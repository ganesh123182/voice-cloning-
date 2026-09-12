package com.voice.shield

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import android.util.Log

/**
 * BroadcastReceiver that listens for phone state changes.
 * Starts LiveCallService when a call is answered (OFFHOOK)
 * and stops it when the call ends (IDLE).
 */
class CallReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "CallReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE) ?: return
        Log.d(TAG, "Phone state changed: $state")

        when (state) {
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                // Call answered (incoming or outgoing)
                Log.i(TAG, "Call answered: Starting LiveCallService")
                startAudioService(context)
            }
            TelephonyManager.EXTRA_STATE_IDLE -> {
                // Call ended or rejected - check that device is actually idle
                val tm = context.getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
                @Suppress("DEPRECATION")
                val currentCallState = tm?.callState
                if (currentCallState != null && currentCallState != TelephonyManager.CALL_STATE_IDLE) {
                    Log.i(TAG, "Ignoring IDLE broadcast: device callState is active ($currentCallState)")
                    return
                }
                Log.i(TAG, "Call ended: Stopping LiveCallService")
                stopAudioService(context)
            }
            TelephonyManager.EXTRA_STATE_RINGING -> {
                Log.d(TAG, "Incoming call ringing...")
            }
        }
    }

    private fun startAudioService(context: Context) {
        val serviceIntent = Intent(context, LiveCallService::class.java)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }

    private fun stopAudioService(context: Context) {
        val serviceIntent = Intent(context, LiveCallService::class.java)
        context.stopService(serviceIntent)
    }
}
