package com.voice.shield

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.text.TextUtils
import android.view.View
import android.widget.Chronometer
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MockCallActivity : AppCompatActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            checkOverlayPermission()
        } else {
            Toast.makeText(this, "Microphone permission is required for detection.", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        checkAccessibilityService()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_mock_call)

        // Hide action bar for full screen effect
        supportActionBar?.hide()

        // Setup the end call button
        findViewById<View>(R.id.btn_end_call).setOnClickListener {
            // Stop recording
            val stopIntent = Intent(LiveCallService.ACTION_STOP_RECORDING).apply {
                setPackage(packageName)
            }
            sendBroadcast(stopIntent)
            
            // Close the mock call screen
            finish()
        }

        // Check permissions before starting
        checkMicrophonePermission()
    }

    private fun checkMicrophonePermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            checkOverlayPermission()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun checkOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            AlertDialog.Builder(this)
                .setTitle("Overlay Permission Required")
                .setMessage("To show the Live Analysis HUD during a call, please allow the app to display over other apps.")
                .setPositiveButton("Open Settings") { _, _ ->
                    val intent = Intent(
                        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                    overlayPermissionLauncher.launch(intent)
                }
                .setNegativeButton("Cancel") { _, _ ->
                    Toast.makeText(this, "HUD won't be visible", Toast.LENGTH_SHORT).show()
                    checkAccessibilityService()
                }
                .show()
        } else {
            checkAccessibilityService()
        }
    }

    private fun checkAccessibilityService() {
        if (isAccessibilityServiceEnabled(this, LiveCallService::class.java)) {
            startCallSimulation()
        } else {
            AlertDialog.Builder(this)
                .setTitle("Accessibility Service Required")
                .setMessage("TrustVoice needs its Accessibility Service enabled to monitor live calls and show the HUD. Please enable 'TrustVoice' in Settings.")
                .setPositiveButton("Open Settings") { _, _ ->
                    val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                    startActivity(intent)
                    // We'll finish the mock call for now. When they enable it, they can try again.
                    Toast.makeText(this, "Try simulating call again after enabling.", Toast.LENGTH_LONG).show()
                    finish()
                }
                .setNegativeButton("Cancel") { _, _ ->
                    finish()
                }
                .show()
        }
    }

    private fun startCallSimulation() {
        // Start the call timer
        val chronometer = findViewById<Chronometer>(R.id.chronometer_call_time)
        chronometer.base = SystemClock.elapsedRealtime()
        chronometer.start()

        // Tell the LiveCallService to start recording (simulating a real call offhook)
        val startIntent = Intent(LiveCallService.ACTION_START_RECORDING).apply {
            setPackage(packageName)
        }
        sendBroadcast(startIntent)
    }

    override fun onDestroy() {
        super.onDestroy()
        // Ensure recording stops if user presses back button to exit
        val stopIntent = Intent(LiveCallService.ACTION_STOP_RECORDING).apply {
            setPackage(packageName)
        }
        sendBroadcast(stopIntent)
    }

    private fun isAccessibilityServiceEnabled(context: Context, service: Class<*>): Boolean {
        var accessibilityEnabled = 0
        try {
            accessibilityEnabled = Settings.Secure.getInt(
                context.contentResolver,
                Settings.Secure.ACCESSIBILITY_ENABLED
            )
        } catch (e: Settings.SettingNotFoundException) {
            // Setting not found
        }
        
        if (accessibilityEnabled == 1) {
            val settingValue = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            )
            if (settingValue != null) {
                val ms = TextUtils.SimpleStringSplitter(':')
                ms.setString(settingValue)
                while (ms.hasNext()) {
                    val accessibilityService = ms.next()
                    if (accessibilityService.contains(service.name)) {
                        return true
                    }
                }
            }
        }
        return false
    }
}
