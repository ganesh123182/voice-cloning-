package com.voice.shield

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var tvPermMic: TextView
    private lateinit var tvPermPhone: TextView
    private lateinit var tvPermOverlay: TextView
    private lateinit var etServerUrl: EditText
    private lateinit var btnGrantPermissions: Button
    private lateinit var btnSaveUrl: Button
    private lateinit var btnToggleLiveService: Button
    private lateinit var btnTestHud: Button
    private lateinit var btnSimulateScam: Button
    private var isLiveServiceRunning = false

    private val prefs by lazy {
        getSharedPreferences("TrustVoicePrefs", Context.MODE_PRIVATE)
    }

    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        updatePermissionStatus()
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            Toast.makeText(this, "Standard permissions granted!", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Some permissions were denied", Toast.LENGTH_SHORT).show()
        }
    }

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        updatePermissionStatus()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind Views
        tvPermMic = findViewById(R.id.tv_perm_mic)
        tvPermPhone = findViewById(R.id.tv_perm_phone)
        tvPermOverlay = findViewById(R.id.tv_perm_overlay)
        etServerUrl = findViewById(R.id.et_server_url)
        btnGrantPermissions = findViewById(R.id.btn_grant_permissions)
        btnSaveUrl = findViewById(R.id.btn_save_url)
        btnToggleLiveService = findViewById(R.id.btn_toggle_live_service)
        btnTestHud = findViewById(R.id.btn_test_hud)
        btnSimulateScam = findViewById(R.id.btn_simulate_scam)

        // Load saved URL or default
        val defaultUrl = "ws://10.163.249.212:8000/api/monitoring/live?token=test_token"
        val savedUrl = prefs.getString("websocket_url", defaultUrl)
        etServerUrl.setText(savedUrl)

        // Listeners
        btnSaveUrl.setOnClickListener {
            val url = etServerUrl.text.toString().trim()
            if (url.isNotEmpty()) {
                prefs.edit().putString("websocket_url", url).apply()
                Toast.makeText(this, "Server URL saved!", Toast.LENGTH_SHORT).show()
            }
        }

        btnGrantPermissions.setOnClickListener {
            requestAllPermissions()
        }

        btnToggleLiveService.setOnClickListener {
            if (!isLiveServiceRunning) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
                    Toast.makeText(this, "Please grant Overlay permission first!", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                val serviceIntent = Intent(this, LiveCallService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent)
                } else {
                    startService(serviceIntent)
                }
                isLiveServiceRunning = true
                btnToggleLiveService.text = "Stop Live Call Monitor"
                btnToggleLiveService.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#DC2626"))
                Toast.makeText(this, "Live Call Monitor started! Streaming mic to AI server.", Toast.LENGTH_SHORT).show()
            } else {
                val serviceIntent = Intent(this, LiveCallService::class.java)
                stopService(serviceIntent)
                isLiveServiceRunning = false
                btnToggleLiveService.text = "Start Live Call Monitor"
                btnToggleLiveService.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#16A34A"))
                Toast.makeText(this, "Live Call Monitor stopped.", Toast.LENGTH_SHORT).show()
            }
        }

        btnTestHud.setOnClickListener {
            testFloatingHUD(isFake = false)
        }

        btnSimulateScam.setOnClickListener {
            testFloatingHUD(isFake = true)
        }
    }

    override fun onResume() {
        super.onResume()
        updatePermissionStatus()
    }

    private fun updatePermissionStatus() {
        val micGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        val phoneGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.READ_PHONE_STATE
        ) == PackageManager.PERMISSION_GRANTED

        val overlayGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Settings.canDrawOverlays(this)
        } else {
            true
        }

        tvPermMic.text = if (micGranted) "✅ Microphone (AudioRecord): Granted" else "❌ Microphone: NOT Granted"
        tvPermMic.setTextColor(if (micGranted) Color.parseColor("#22C55E") else Color.parseColor("#EF4444"))

        tvPermPhone.text = if (phoneGranted) "✅ Phone State (Call Detect): Granted" else "❌ Phone State: NOT Granted"
        tvPermPhone.setTextColor(if (phoneGranted) Color.parseColor("#22C55E") else Color.parseColor("#EF4444"))

        tvPermOverlay.text = if (overlayGranted) "✅ Display Over Other Apps: Granted" else "❌ Display Over Other Apps: NOT Granted"
        tvPermOverlay.setTextColor(if (overlayGranted) Color.parseColor("#22C55E") else Color.parseColor("#EF4444"))
    }

    private fun requestAllPermissions() {
        val permissionsToRequest = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissionsToRequest.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        requestPermissionsLauncher.launch(permissionsToRequest.toTypedArray())

        // Overlay permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
        }
    }

    private fun testFloatingHUD(isFake: Boolean) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Please enable 'Display over other apps' first!", Toast.LENGTH_LONG).show()
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
            return
        }

        // Start Floating HUD service
        val hudIntent = Intent(this, FloatingHUDService::class.java)
        startService(hudIntent)

        // Post simulated analysis update after brief pause
        window.decorView.postDelayed({
            val jsonPayload = if (isFake) {
                JSONObject().apply {
                    put("speaker", "caller")
                    put("is_fake", true)
                    put("risk_score", 94.5)
                    put("threat_advisory", "🚨 WARNING: Synthetic AI Voice Cloning Detected! Do NOT transfer funds.")
                }.toString()
            } else {
                JSONObject().apply {
                    put("speaker", "user")
                    put("is_fake", false)
                    put("risk_score", 4.2)
                    put("threat_advisory", "Voice identity verified. Call is safe.")
                }.toString()
            }

            val broadcastIntent = Intent(LiveCallService.ACTION_HUD_UPDATE).apply {
                putExtra(LiveCallService.EXTRA_JSON_PAYLOAD, jsonPayload)
                setPackage(packageName)
            }
            sendBroadcast(broadcastIntent)
        }, 500)
    }
}
