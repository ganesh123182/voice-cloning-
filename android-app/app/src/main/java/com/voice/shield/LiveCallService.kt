package com.voice.shield

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground Service that captures live audio from the microphone during
 * an active phone call, streams 3-second PCM chunks over a WebSocket to
 * the TrustVoice backend, and broadcasts the server's JSON analysis
 * results as local broadcasts for the Floating HUD overlay.
 *
 * Lifecycle:
 *   CallReceiver.OFFHOOK  →  startForegroundService(LiveCallService)
 *   CallReceiver.IDLE     →  stopService(LiveCallService)
 */
class LiveCallService : Service() {

    companion object {
        private const val TAG = "LiveCallService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "TrustVoiceCallShieldChannel"

        // Broadcast action and extra key for the Floating HUD
        const val ACTION_HUD_UPDATE = "com.voice.shield.HUD_UPDATE"
        const val EXTRA_JSON_PAYLOAD = "json_payload"

        // ---- Audio Configuration ----
        private const val SAMPLE_RATE = 16000          // 16 kHz
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        // 3-second buffer: 16000 samples/s × 2 bytes/sample × 1 channel × 3 s = 96,000 bytes
        private const val CHUNK_DURATION_SEC = 3
        private const val BUFFER_SIZE_BYTES = SAMPLE_RATE * 2 * 1 * CHUNK_DURATION_SEC  // 96000

        // PC's IP on phone's hotspot Wi-Fi network
        private const val DEFAULT_WEBSOCKET_URL =
            "ws://10.163.249.212:8000/api/monitoring/live?token=test_token"

        // Reconnection parameters
        private const val MAX_RECONNECT_ATTEMPTS = 5
        private const val RECONNECT_DELAY_MS = 3000L
    }

    // ---- State ----
    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)

    private var webSocket: WebSocket? = null
    private val isWebSocketConnected = AtomicBoolean(false)
    private var reconnectAttempts = 0

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MINUTES)   // keep-alive for streaming
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    // =========================================================================
    //  Service lifecycle
    // =========================================================================

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "Starting LiveCallService")

        val notification = buildNotification()

        // Android 10 (Q)+ supports typed foreground services; Android 14 (U)
        // mandates FOREGROUND_SERVICE_MICROPHONE permission for this type.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        // Start the Floating HUD safely now that we are in the foreground
        val hudIntent = Intent(this, FloatingHUDService::class.java)
        startService(hudIntent)

        connectWebSocket()
        startAudioCapture()

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null  // Not a bound service

    override fun onDestroy() {
        Log.i(TAG, "Destroying LiveCallService")
        
        // Stop the Floating HUD
        val hudIntent = Intent(this, FloatingHUDService::class.java)
        stopService(hudIntent)
        
        stopAudioCapture()
        disconnectWebSocket()
        serviceScope.cancel()
        serviceJob.cancel()
        Log.i(TAG, "LiveCallService destroyed cleanly")
        super.onDestroy()
    }

    // =========================================================================
    //  Notification
    // =========================================================================

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "TrustVoice Call Shield",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Active call monitoring for deepfake detection"
                setShowBadge(false)
            }
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val pending = PendingIntent.getActivity(
            this, 0, launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("TrustVoice Call Shield Active")
            .setContentText("Monitoring live call for deepfakes…")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pending)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    // =========================================================================
    //  WebSocket
    // =========================================================================

    private fun connectWebSocket() {
        val prefs = getSharedPreferences("TrustVoicePrefs", Context.MODE_PRIVATE)
        val url = prefs.getString("websocket_url", null) ?: DEFAULT_WEBSOCKET_URL
        Log.i(TAG, "Connecting WebSocket to: $url")
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected -> HTTP ${response.code} Switching Protocols")
                isWebSocketConnected.set(true)
                reconnectAttempts = 0
            }

            override fun onMessage(ws: WebSocket, text: String) {
                Log.d(TAG, "WebSocket received JSON: $text")
                handleServerResponse(text)
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closing: $code / $reason")
                ws.close(code, reason)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $code / $reason")
                isWebSocketConnected.set(false)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                isWebSocketConnected.set(false)
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            Log.w(TAG, "Max WebSocket reconnection attempts reached ($MAX_RECONNECT_ATTEMPTS)")
            return
        }
        reconnectAttempts++
        serviceScope.launch {
            val backoff = RECONNECT_DELAY_MS * reconnectAttempts
            Log.i(TAG, "Reconnecting WebSocket in ${backoff}ms (attempt $reconnectAttempts)")
            delay(backoff)
            if (isActive && isRecording.get()) {
                connectWebSocket()
            }
        }
    }

    private fun disconnectWebSocket() {
        webSocket?.close(1000, "Service destroyed")
        webSocket = null
        isWebSocketConnected.set(false)
    }

    // =========================================================================
    //  Audio Capture
    // =========================================================================

    @Suppress("MissingPermission")  // Permission checked at runtime in the Activity
    private fun startAudioCapture() {
        if (isRecording.get()) {
            Log.w(TAG, "Audio capture already running – ignoring duplicate start")
            return
        }

        try {
            val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBuf == AudioRecord.ERROR || minBuf == AudioRecord.ERROR_BAD_VALUE) {
                Log.e(TAG, "AudioRecord.getMinBufferSize returned error: $minBuf")
                return
            }

            // Attempt to use VOICE_DOWNLINK (for clear incoming audio), fallback to MIC
            val audioSource = MediaRecorder.AudioSource.VOICE_COMMUNICATION
            
            audioRecord = AudioRecord(
                audioSource,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                maxOf(minBuf, BUFFER_SIZE_BYTES)
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize (state=${audioRecord?.state})")
                audioRecord?.release()
                audioRecord = null
                return
            }

            audioRecord!!.startRecording()
            isRecording.set(true)
            Log.i(TAG, "AudioRecord started at ${SAMPLE_RATE}Hz")

            // Launch the continuous read-send coroutine
            serviceScope.launch { audioReadLoop() }

        } catch (e: SecurityException) {
            Log.e(TAG, "RECORD_AUDIO permission not granted!", e)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio capture", e)
        }
    }

    /**
     * Continuously reads from the AudioRecord in a background coroutine,
     * accumulates exactly [BUFFER_SIZE_BYTES] (3 seconds), and pushes
     * the chunk over the WebSocket as a raw binary message.
     */
    private suspend fun audioReadLoop() {
        val buffer = ByteArray(BUFFER_SIZE_BYTES)
        var offset = 0

        while (isRecording.get() && serviceScope.isActive) {
            val remaining = BUFFER_SIZE_BYTES - offset
            val read = audioRecord?.read(buffer, offset, remaining) ?: break

            when {
                read > 0 -> {
                    offset += read
                    if (offset >= BUFFER_SIZE_BYTES) {
                        // Full 3-second chunk ready
                        sendAudioChunk(buffer.copyOf())   // copy to avoid mutation
                        offset = 0
                    }
                }
                read == AudioRecord.ERROR_INVALID_OPERATION -> {
                    Log.e(TAG, "AudioRecord read: ERROR_INVALID_OPERATION")
                    break
                }
                read == AudioRecord.ERROR_BAD_VALUE -> {
                    Log.e(TAG, "AudioRecord read: ERROR_BAD_VALUE")
                    break
                }
                read == AudioRecord.ERROR_DEAD_OBJECT -> {
                    Log.e(TAG, "AudioRecord read: ERROR_DEAD_OBJECT – microphone lost")
                    break
                }
                else -> {
                    // read == 0 → no data available yet, keep looping
                }
            }
        }

        Log.d(TAG, "Audio read loop exited")
    }

    /**
     * Sends a raw PCM byte array over the WebSocket.
     * Logs the exact byte count for Logcat verification.
     */
    private fun sendAudioChunk(chunk: ByteArray) {
        if (!isWebSocketConnected.get()) {
            Log.w(TAG, "WebSocket not connected – dropping ${chunk.size} byte chunk")
            return
        }

        val sent = webSocket?.send(chunk.toByteString()) ?: false
        if (sent) {
            Log.i(TAG, "Sent ${chunk.size} bytes over WebSocket -> HTTP 101 Switching Protocols")
        } else {
            Log.w(TAG, "WebSocket send failed for ${chunk.size} bytes (queue full or closed)")
        }
    }

    private fun stopAudioCapture() {
        isRecording.set(false)
        try {
            audioRecord?.stop()
        } catch (e: IllegalStateException) {
            Log.w(TAG, "AudioRecord.stop() threw: ${e.message}")
        }
        audioRecord?.release()
        audioRecord = null
        Log.i(TAG, "AudioRecord stopped and released")
    }

    // =========================================================================
    //  Server Response Handling → Floating HUD Broadcast
    // =========================================================================

    /**
     * Parses the JSON payload from the backend and fires a local broadcast
     * so the Floating HUD overlay can display risk information.
     *
     * Expected JSON fields: speaker, risk_score, label, is_alert, suggestion
     */
    private fun handleServerResponse(jsonText: String) {
        try {
            val json = JSONObject(jsonText)

            // Log key fields for diagnostics
            val speaker = json.optString("speaker", "unknown")
            val risk = json.optDouble("risk_score", 0.0)
            val label = json.optString("label", "")
            Log.d(TAG, "Analysis → speaker=$speaker  risk=$risk  label=$label")

            // Broadcast to HUD
            val broadcast = Intent(ACTION_HUD_UPDATE).apply {
                putExtra(EXTRA_JSON_PAYLOAD, jsonText)
                // On Android 14+, specify the package so the broadcast is
                // delivered only within this app (implicit broadcast restriction).
                setPackage(packageName)
            }
            sendBroadcast(broadcast)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse server response", e)
        }
    }
}
