package com.voice.shield

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
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
        @Suppress("DEPRECATION")
        val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER) ?: "Unknown"
        Log.d(TAG, "Phone state changed: $state, number: $incomingNumber")

        when (state) {
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                // Call answered (incoming or outgoing)
                val callerName = getContactName(context, incomingNumber)
                Log.i(TAG, "Call answered: Sending START_RECORDING command to LiveCallService. Caller: $callerName ($incomingNumber)")
                
                val commandIntent = Intent(LiveCallService.ACTION_START_RECORDING).apply {
                    setPackage(context.packageName)
                    putExtra("caller_number", incomingNumber)
                    putExtra("caller_name", callerName)
                }
                context.sendBroadcast(commandIntent)
            }
            TelephonyManager.EXTRA_STATE_IDLE -> {
                // Call ended or rejected - check that device is actually idle
                val tm = context.getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
                @Suppress("DEPRECATION")
                val currentCallState = tm?.callState
                if ((currentCallState != null) && (currentCallState != TelephonyManager.CALL_STATE_IDLE)) {
                    Log.i(TAG, "Ignoring IDLE broadcast: device callState is active ($currentCallState)")
                    return
                }
                Log.i(TAG, "Call ended: Sending STOP_RECORDING command to LiveCallService")
                val commandIntent = Intent(LiveCallService.ACTION_STOP_RECORDING).apply {
                    setPackage(context.packageName)
                }
                context.sendBroadcast(commandIntent)
            }
            TelephonyManager.EXTRA_STATE_RINGING -> {
                Log.d(TAG, "Incoming call ringing... Number: $incomingNumber")
            }
        }
    }

    private fun getContactName(context: Context, phoneNumber: String): String {
        if (phoneNumber == "Unknown" || phoneNumber.isBlank()) return "Unknown Caller"
        
        var contactName = "Unknown Caller"
        try {
            val uri = Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, Uri.encode(phoneNumber))
            val projection = arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME)
            
            context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIndex = cursor.getColumnIndex(ContactsContract.PhoneLookup.DISPLAY_NAME)
                    if (nameIndex >= 0) {
                        contactName = cursor.getString(nameIndex)
                    }
                }
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "READ_CONTACTS permission not granted", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error looking up contact name", e)
        }
        return contactName
    }
}
