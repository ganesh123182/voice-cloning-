package com.voice.shield

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.accessibilityservice.AccessibilityService
import android.content.BroadcastReceiver
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.view.accessibility.AccessibilityEvent
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
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
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
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
class LiveCallService : AccessibilityService() {

    companion object {
        private const val TAG = "LiveCallService"
        private const val CHANNEL_ID = "TrustVoiceCallShieldChannel"
        private const val NOTIFICATION_ID = 2001

        // Broadcast action and extra key for the Floating HUD
        const val ACTION_HUD_UPDATE = "com.voice.shield.HUD_UPDATE"
        const val EXTRA_JSON_PAYLOAD = "json_payload"
        
        // Commands received from CallReceiver
        const val ACTION_START_RECORDING = "com.voice.shield.START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.voice.shield.STOP_RECORDING"

        // ---- Audio Configuration ----
        private const val SAMPLE_RATE = 16000          // 16 kHz
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        // 1.5-second sliding window buffer: 16000 samples/s × 2 bytes/sample × 1 channel × 1.5 s = 48,000 bytes
        private const val CHUNK_DURATION_MS = 1500
        private const val BUFFER_SIZE_BYTES = 48000

        // PC's IP on phone's hotspot / Wi-Fi network
        private const val DEFAULT_WEBSOCKET_URL =
            "ws://10.78.43.212:8000/api/monitoring/live?token=test_token"

        // Reconnection parameters
        private const val MAX_RECONNECT_ATTEMPTS = 5
        private const val RECONNECT_DELAY_MS = 3000L
    }

    // ---- State ----
    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)

    private var activeAudioSourceIndex = 0
    private var audioSourcesToTry = intArrayOf()

    private var webSocket: WebSocket? = null
    private val isWebSocketConnected = AtomicBoolean(false)
    private var reconnectAttempts = 0

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MINUTES)   // keep-alive for streaming
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    private var currentCallerNumber = "Unknown"
    private var currentCallerName = "Unknown Caller"
    private var isSimulation = false

    private fun getAudioSourceName(source: Int): String {
        return when (source) {
            MediaRecorder.AudioSource.MIC -> "MIC"
            MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "VOICE_COMMUNICATION"
            MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
            MediaRecorder.AudioSource.CAMCORDER -> "CAMCORDER"
            MediaRecorder.AudioSource.DEFAULT -> "DEFAULT"
            else -> "SOURCE_$source"
        }
    }

    private fun buildNotification(callerName: String, callerNumber: String): Notification {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val callerDisplay = if (callerName == "Unknown Caller") callerNumber else "$callerName ($callerNumber)"
        val contentText = if (isSimulation) {
            "Simulating live call monitoring for AI deepfakes..."
        } else {
            "Active call with $callerDisplay — monitoring for AI voice scams (Use Speakerphone)"
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("TrustVoice Call Shield Active")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private val commandReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val action = intent?.action
            Log.i(TAG, "CommandReceiver received action: $action")
            
            if (action == ACTION_START_RECORDING) {
                if (!isRecording.get()) {
                    currentCallerNumber = intent?.getStringExtra("caller_number") ?: "Unknown"
                    currentCallerName = intent?.getStringExtra("caller_name") ?: "Unknown Caller"
                    isSimulation = intent?.getBooleanExtra("is_simulation", false) ?: false
                    
                    // Promote to foreground service with microphone type to satisfy Android background audio capture policies
                    try {
                        val notification = buildNotification(currentCallerName, currentCallerNumber)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            startForeground(
                                NOTIFICATION_ID,
                                notification,
                                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                            )
                        } else {
                            startForeground(NOTIFICATION_ID, notification)
                        }
                        Log.i(TAG, "Promoted LiveCallService to foreground with MICROPHONE type")
                    } catch (e: Exception) {
                        Log.w(TAG, "startForeground threw exception: ${e.message}")
                    }

                    val hudIntent = Intent(context, FloatingHUDService::class.java)
                    startService(hudIntent)
                    
                    connectWebSocket()
                    startAudioCapture()
                }
            } else if (action == ACTION_STOP_RECORDING) {
                if (isRecording.get()) {
                    stopAudioCapture()
                    disconnectWebSocket()
                    
                    val hudIntent = Intent(context, FloatingHUDService::class.java)
                    stopService(hudIntent)

                    try {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                            stopForeground(STOP_FOREGROUND_REMOVE)
                        } else {
                            @Suppress("DEPRECATION")
                            stopForeground(true)
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "stopForeground threw exception: ${e.message}")
                    }
                }
            }
        }
    }

    // =========================================================================
    //  AccessibilityService lifecycle
    // =========================================================================

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // We do not process UI accessibility events, we just need the background privilege
    }

    override fun onInterrupt() {
        Log.w(TAG, "AccessibilityService interrupted")
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.i(TAG, "AccessibilityService onServiceConnected")

        // Register receiver for CallReceiver commands
        val filter = IntentFilter().apply {
            addAction(ACTION_START_RECORDING)
            addAction(ACTION_STOP_RECORDING)
        }
        
        ContextCompat.registerReceiver(this, commandReceiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        Log.i(TAG, "AccessibilityService running, waiting for phone call to begin recording...")
    }

    override fun onDestroy() {
        Log.i(TAG, "Destroying LiveCallService")
        
        try {
            unregisterReceiver(commandReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister receiver", e)
        }
        
        // Stop the Floating HUD
        val hudIntent = Intent(this, FloatingHUDService::class.java)
        stopService(hudIntent)
        
        stopAudioCapture()
        disconnectWebSocket()

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
        } catch (e: Exception) {
            // ignore
        }

        serviceScope.cancel()
        serviceJob.cancel()
        Log.i(TAG, "LiveCallService destroyed cleanly")
        super.onDestroy()
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "TrustVoice Call Shield",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Active call monitoring for deepfake detection"
            setShowBadge(false)
        }
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(channel)
    }

    // =========================================================================
    //  WebSocket
    // =========================================================================

    private fun connectWebSocket() {
        val prefs = getSharedPreferences("TrustVoicePrefs", MODE_PRIVATE)
        val authPrefs = getSharedPreferences("AuthPrefs", MODE_PRIVATE)
        val authToken = authPrefs.getString("jwt_token", null)
            ?: authPrefs.getString("user_id", null)
            ?: "demo_user"

        val baseUrl = prefs.getString("websocket_url", null) ?: DEFAULT_WEBSOCKET_URL
        
        // Convert ws:// -> http:// temporarily so HttpUrl can safely parse query parameters
        val httpUrlStr = when {
            baseUrl.startsWith("ws://") -> "http://" + baseUrl.removePrefix("ws://")
            baseUrl.startsWith("wss://") -> "https://" + baseUrl.removePrefix("wss://")
            else -> baseUrl
        }
        val httpUrl = httpUrlStr.toHttpUrlOrNull()
        val url = if (httpUrl != null) {
            val builder = httpUrl.newBuilder()
                .addQueryParameter("caller_number", currentCallerNumber)
                .addQueryParameter("caller_name", currentCallerName)
                .addQueryParameter("is_simulation", isSimulation.toString())
            if (httpUrl.queryParameter("token") == null) {
                builder.addQueryParameter("token", authToken)
            }
            val resultHttp = builder.build().toString()
            when {
                baseUrl.startsWith("ws://") -> "ws://" + resultHttp.removePrefix("http://")
                baseUrl.startsWith("wss://") -> "wss://" + resultHttp.removePrefix("https://")
                else -> resultHttp
            }
        } else {
            val separator = if (baseUrl.contains("?")) "&" else "?"
            "$baseUrl${separator}caller_number=${java.net.URLEncoder.encode(currentCallerNumber, "UTF-8")}&caller_name=${java.net.URLEncoder.encode(currentCallerName, "UTF-8")}&is_simulation=$isSimulation&token=${java.net.URLEncoder.encode(authToken, "UTF-8")}"
        }
        
        Log.i(TAG, "Connecting WebSocket to: $url")
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected -> HTTP ${response.code} Switching Protocols")
                isWebSocketConnected.set(true)
                reconnectAttempts = 0
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.i(TAG, "WebSocket received JSON: $text")
                handleServerResponse(text)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closing: $code / $reason")
                webSocket.close(code, reason)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $code / $reason")
                isWebSocketConnected.set(false)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
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

    @Suppress("MissingPermission")
    private fun initAudioRecordForSource(sourceIndex: Int): AudioRecord? {
        if (sourceIndex < 0 || sourceIndex >= audioSourcesToTry.size) return null
        val src = audioSourcesToTry[sourceIndex]
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBuf == AudioRecord.ERROR || minBuf == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "AudioRecord.getMinBufferSize returned error: $minBuf")
            return null
        }
        return try {
            val record = AudioRecord(
                src,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                maxOf(minBuf, BUFFER_SIZE_BYTES)
            )
            if (record.state == AudioRecord.STATE_INITIALIZED) {
                record
            } else {
                record.release()
                null
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed initializing ${getAudioSourceName(src)}: ${e.message}")
            null
        }
    }

    @Suppress("MissingPermission")  // Permission checked at runtime in the Activity
    private fun startAudioCapture() {
        if (isRecording.get()) {
            Log.w(TAG, "Audio capture already running – ignoring duplicate start")
            return
        }

        try {
            audioSourcesToTry = if (isSimulation) {
                intArrayOf(
                    MediaRecorder.AudioSource.MIC,
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    MediaRecorder.AudioSource.DEFAULT
                )
            } else {
                // For real phone calls, Android telephony HAL silences standard MIC (returns 0s).
                // VOICE_COMMUNICATION is designed for active call audio with hardware AEC/NS.
                // VOICE_RECOGNITION bypasses telecom mute on many OEM devices.
                // CAMCORDER uses the secondary exterior mic.
                // MIC & DEFAULT as final fallbacks.
                intArrayOf(
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    MediaRecorder.AudioSource.MIC,
                    MediaRecorder.AudioSource.CAMCORDER,
                    MediaRecorder.AudioSource.DEFAULT
                )
            }

            activeAudioSourceIndex = 0
            var initializedRecord: AudioRecord? = null

            for (i in audioSourcesToTry.indices) {
                val record = initAudioRecordForSource(i)
                if (record != null) {
                    initializedRecord = record
                    activeAudioSourceIndex = i
                    break
                }
            }

            if (initializedRecord == null) {
                Log.e(TAG, "Failed to initialize AudioRecord with any available audio source")
                return
            }

            audioRecord = initializedRecord
            audioRecord!!.startRecording()
            isRecording.set(true)
            val sourceName = getAudioSourceName(audioSourcesToTry[activeAudioSourceIndex])
            Log.i(TAG, "AudioRecord started at ${SAMPLE_RATE}Hz using source: $sourceName (isSimulation=$isSimulation)")

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
     * accumulates exactly [BUFFER_SIZE_BYTES] (1.5 seconds / 48000 bytes),
     * checks for silent (all-zero) buffers, dynamically falls back to alternative
     * sources if the OS silences the current source, and pushes valid chunks
     * over the WebSocket as raw binary messages.
     */
    private fun audioReadLoop() {
        val buffer = ByteArray(BUFFER_SIZE_BYTES)
        var offset = 0
        var consecutiveAllZeroChunks = 0

        while (isRecording.get() && serviceScope.isActive) {
            val remaining = BUFFER_SIZE_BYTES - offset
            val currentRec = audioRecord ?: break
            val read = try {
                currentRec.read(buffer, offset, remaining)
            } catch (e: Exception) {
                Log.e(TAG, "AudioRecord read exception: ${e.message}")
                break
            }

            when {
                read > 0 -> {
                    offset += read
                    if (offset >= BUFFER_SIZE_BYTES) {
                        // Check if this chunk is completely zero (hardware/OS silenced)
                        var isAllZero = true
                        for (b in buffer) {
                            if (b != 0.toByte()) {
                                isAllZero = false
                                break
                            }
                        }

                        if (isAllZero) {
                            consecutiveAllZeroChunks++
                            val currentSource = getAudioSourceName(audioSourcesToTry[activeAudioSourceIndex])
                            Log.w(TAG, "Chunk was 100% zeros! (silent chunk #$consecutiveAllZeroChunks on $currentSource)")

                            // If we get 2 consecutive silent chunks (3 seconds of pure 0x00) and have more sources:
                            if (consecutiveAllZeroChunks >= 2 && activeAudioSourceIndex + 1 < audioSourcesToTry.size) {
                                activeAudioSourceIndex++
                                val nextSource = getAudioSourceName(audioSourcesToTry[activeAudioSourceIndex])
                                Log.i(TAG, "Auto-switching from $currentSource to $nextSource to bypass OS silence...")

                                try {
                                    audioRecord?.stop()
                                } catch (e: Exception) { /* ignore */ }
                                audioRecord?.release()
                                audioRecord = null

                                val newRecord = initAudioRecordForSource(activeAudioSourceIndex)
                                if (newRecord != null) {
                                    audioRecord = newRecord
                                    audioRecord!!.startRecording()
                                    consecutiveAllZeroChunks = 0
                                    Log.i(TAG, "Successfully switched to $nextSource")
                                } else {
                                    Log.w(TAG, "Failed initializing $nextSource, keeping search open")
                                }
                            }
                        } else {
                            // Non-zero audio detected! Reset silence counter
                            if (consecutiveAllZeroChunks > 0) {
                                Log.i(TAG, "Real acoustic audio detected on source ${getAudioSourceName(audioSourcesToTry[activeAudioSourceIndex])}!")
                                consecutiveAllZeroChunks = 0
                            }
                        }

                        // Send audio chunk to backend (copy to avoid mutation)
                        sendAudioChunk(buffer.copyOf())
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
        } catch (e: Exception) {
            Log.w(TAG, "AudioRecord.stop() threw: ${e.message}")
        } finally {
            audioRecord?.release()
            audioRecord = null
            Log.i(TAG, "AudioRecord stopped and released")
        }
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

            // Log key fields for diagnostics (Log.i to ensure visibility on production devices)
            val speaker = json.optString("speaker", "unknown")
            val risk = json.optDouble("risk_score", 0.0)
            val label = json.optString("label", "")
            Log.i(TAG, "Analysis → speaker=$speaker  risk=$risk  label=$label")

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
