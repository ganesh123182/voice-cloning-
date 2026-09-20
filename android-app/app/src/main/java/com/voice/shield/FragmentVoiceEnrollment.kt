package com.voice.shield

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.voice.shield.api.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile

class FragmentVoiceEnrollment : Fragment(R.layout.fragment_voice_enrollment) {

    private lateinit var btnStartRecording: Button
    private lateinit var progressEnrollment: ProgressBar
    private lateinit var txtRecordingTimer: TextView
    private var btnBack: View? = null
    
    private var cardEnrolledHash: View? = null
    private var txtVoiceHashDisplay: TextView? = null
    private var txtEnrollmentIdDisplay: TextView? = null
    private var btnCopyHash: Button? = null
    
    private var audioRecord: AudioRecord? = null
    private var audioFile: File? = null
    private val isRecording = java.util.concurrent.atomic.AtomicBoolean(false)
    private var recordingJob: Job? = null
    private var writeJob: Job? = null
    private var timerJob: Job? = null
    private var recordingStartTime = 0L
    
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                // Permission granted
            } else {
                Toast.makeText(requireContext(), "Microphone permission required", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        btnStartRecording = view.findViewById(R.id.btn_start_recording)
        progressEnrollment = view.findViewById(R.id.progress_enrollment)
        txtRecordingTimer = view.findViewById(R.id.txt_recording_timer)
        btnBack = view.findViewById(R.id.btn_back_enrollment)

        cardEnrolledHash = view.findViewById(R.id.card_enrolled_hash)
        txtVoiceHashDisplay = view.findViewById(R.id.txt_voice_hash_display)
        txtEnrollmentIdDisplay = view.findViewById(R.id.txt_enrollment_id_display)
        btnCopyHash = view.findViewById(R.id.btn_copy_hash)

        btnBack?.setOnClickListener {
            findNavController().navigateUp()
        }

        btnCopyHash?.setOnClickListener {
            val text = txtVoiceHashDisplay?.text?.toString() ?: ""
            if (text.isNotEmpty() && !text.contains("appear here")) {
                val clipboard = requireContext().getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                val clip = android.content.ClipData.newPlainText("Biometric Voice Hash", text)
                clipboard.setPrimaryClip(clip)
                Toast.makeText(requireContext(), "Voice Hash copied to clipboard!", Toast.LENGTH_SHORT).show()
            }
        }

        // Check local cache for immediate display
        val tvPrefs = requireActivity().getSharedPreferences("TrustVoicePrefs", android.content.Context.MODE_PRIVATE)
        val cachedHash = tvPrefs.getString("enrolled_voice_hash", null)
        val cachedId = tvPrefs.getString("enrolled_id", null)
        if (!cachedHash.isNullOrEmpty()) {
            displayEnrolledVoiceCard(cachedHash, cachedId ?: "Active")
        }

        // Query backend for latest status
        fetchEnrollmentStatus()

        btnStartRecording.setOnClickListener {
            if (isRecording.get()) {
                val elapsedSec = (System.currentTimeMillis() - recordingStartTime) / 1000
                if (elapsedSec < 3) {
                    Toast.makeText(
                        requireContext(),
                        "Please speak for at least 3 seconds (${elapsedSec}s elapsed).",
                        Toast.LENGTH_SHORT
                    ).show()
                    return@setOnClickListener
                }
                stopRecordingAndUpload()
            } else {
                if (checkPermissions()) {
                    startRecording()
                } else {
                    requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
        }
    }

    private fun checkPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            requireContext(),
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return
        
        val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        if (minBufSize <= 0) {
            Toast.makeText(requireContext(), "Microphone not supported", Toast.LENGTH_SHORT).show()
            return
        }

        btnStartRecording.text = "Stop Recording"
        btnStartRecording.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.status_suspicious))
        isRecording.set(true)
        recordingStartTime = System.currentTimeMillis()
        txtRecordingTimer.text = "00:00 / 00:30"
        
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, minBufSize * 2)
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Toast.makeText(requireContext(), "Failed to initialize microphone", Toast.LENGTH_SHORT).show()
            isRecording.set(false)
            resetButton()
            return
        }

        audioFile = File(requireContext().cacheDir, "enrollment_audio.wav")
        audioRecord?.startRecording()

        // Writer coroutine
        writeJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                val out = FileOutputStream(audioFile)
                writeWavHeader(out, 0, 0, sampleRate, 1, sampleRate * 2) // Initial header
                
                val buffer = ByteArray(minBufSize * 2)
                var totalAudioLen = 0L
                while (isRecording.get()) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        out.write(buffer, 0, read)
                        totalAudioLen += read
                    }
                }
                out.flush()
                out.close()
                
                // Finalize accurate RIFF lengths in WAV header
                if (audioFile != null && audioFile!!.exists()) {
                    updateWavHeader(audioFile!!, totalAudioLen)
                }
            } catch (e: Exception) {
                Log.e("Enrollment", "Error writing audio file", e)
            }
        }

        // Timer ticker coroutine
        timerJob = lifecycleScope.launch(Dispatchers.Main) {
            while (isRecording.get()) {
                val elapsed = ((System.currentTimeMillis() - recordingStartTime) / 1000).toInt()
                val min = elapsed / 60
                val sec = elapsed % 60
                txtRecordingTimer.text = String.format("%02d:%02d / 00:30", min, sec)
                delay(500)
            }
        }

        // Auto-stop after 30 seconds
        recordingJob = lifecycleScope.launch {
            delay(30000)
            if (isRecording.get()) {
                stopRecordingAndUpload()
            }
        }
    }

    private fun stopRecordingAndUpload() {
        timerJob?.cancel()
        recordingJob?.cancel()
        isRecording.set(false)

        btnStartRecording.isEnabled = false
        btnStartRecording.text = "Processing..."

        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.e("Enrollment", "Error stopping recorder", e)
        }

        lifecycleScope.launch {
            // Await writer to cleanly finish writing, flushing, and updating WAV header
            writeJob?.join()
            try {
                audioRecord?.release()
            } catch (e: Exception) {}
            audioRecord = null
            uploadAudio()
        }
    }
    
    private fun resetButton() {
        btnStartRecording.isEnabled = true
        btnStartRecording.text = getString(R.string.start_recording)
        btnStartRecording.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.trustvoice_primary))
    }

    private fun uploadAudio() {
        if (audioFile == null || !audioFile!!.exists() || audioFile!!.length() < 1000L) {
            Toast.makeText(requireContext(), "Recording was too short. Please try again.", Toast.LENGTH_SHORT).show()
            resetButton()
            return
        }

        progressEnrollment.visibility = View.VISIBLE

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val authPrefs = requireActivity().getSharedPreferences("AuthPrefs", android.content.Context.MODE_PRIVATE)
                val tvPrefs = requireActivity().getSharedPreferences("TrustVoicePrefs", android.content.Context.MODE_PRIVATE)

                var token = authPrefs.getString("jwt_token", null)
                var userId = authPrefs.getString("user_id", null)
                    ?: tvPrefs.getString("profile_email", null)
                    ?: tvPrefs.getString("profile_name", null)

                if (userId.isNullOrBlank()) {
                    userId = "user_android_" + (System.currentTimeMillis() % 100000)
                    authPrefs.edit().putString("user_id", userId).apply()
                }

                if (token.isNullOrBlank()) {
                    token = "demo_token_" + userId
                    authPrefs.edit().putString("jwt_token", token).apply()
                }

                val requestFile = audioFile!!.asRequestBody("audio/wav".toMediaTypeOrNull())
                val filePart = MultipartBody.Part.createFormData("file", audioFile!!.name, requestFile)
                val userIdPart = userId.toRequestBody("text/plain".toMediaTypeOrNull())

                val response = RetrofitClient.instance.enrollVoice("Bearer $token", filePart, userIdPart)

                withContext(Dispatchers.Main) {
                    progressEnrollment.visibility = View.GONE
                    resetButton()

                    if (response.isSuccessful && response.body() != null) {
                        val respBody = response.body()!!
                        val voiceHash = respBody.voice_hash ?: respBody.sha256_hash ?: respBody.evidence_hash ?: ""
                        val enrollId = respBody.enrollment_id ?: "OK"
                        displayEnrolledVoiceCard(voiceHash, enrollId)
                        Toast.makeText(
                            requireContext(),
                            "Voice Profile Enrolled Successfully!",
                            Toast.LENGTH_SHORT
                        ).show()
                        txtRecordingTimer.text = "00:00 / 00:30"
                    } else {
                        val rawError = response.errorBody()?.string()
                        val errorMsg = try {
                            val json = org.json.JSONObject(rawError ?: "{}")
                            json.optString("detail", "Enrollment failed (${response.code()})")
                        } catch (e: Exception) {
                            "Enrollment failed (${response.code()})"
                        }
                        Toast.makeText(requireContext(), errorMsg, Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressEnrollment.visibility = View.GONE
                    resetButton()
                    Toast.makeText(
                        requireContext(),
                        "Network Error: ${e.localizedMessage ?: "Could not connect"}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
                Log.e("Enrollment", "Error uploading", e)
            }
        }
    }

    private fun displayEnrolledVoiceCard(hash: String, enrollmentId: String) {
        if (hash.isEmpty()) return
        cardEnrolledHash?.visibility = View.VISIBLE
        txtVoiceHashDisplay?.text = hash
        val idDisplay = if (enrollmentId.length > 16) enrollmentId.take(16) + "..." else enrollmentId
        txtEnrollmentIdDisplay?.text = idDisplay
        btnStartRecording.text = "Re-record Voice Profile"

        try {
            val tvPrefs = requireActivity().getSharedPreferences("TrustVoicePrefs", android.content.Context.MODE_PRIVATE)
            tvPrefs.edit()
                .putString("enrolled_voice_hash", hash)
                .putString("enrolled_id", enrollmentId)
                .apply()
        } catch (e: Exception) {
            Log.w("Enrollment", "Could not cache enrolled voice hash", e)
        }
    }

    private fun fetchEnrollmentStatus() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val authPrefs = requireActivity().getSharedPreferences("AuthPrefs", android.content.Context.MODE_PRIVATE)
                val token = authPrefs.getString("jwt_token", null) ?: return@launch
                val resp = RetrofitClient.instance.getEnrollmentStatus("Bearer $token")
                if (resp.isSuccessful && resp.body() != null) {
                    val body = resp.body()!!
                    val hash = body.voice_hash ?: body.evidence_hash
                    withContext(Dispatchers.Main) {
                        displayEnrolledVoiceCard(hash, body.enrollment_id)
                    }
                }
            } catch (e: Exception) {
                // Silently ignore if offline or no enrollment exists yet
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        timerJob?.cancel()
        recordingJob?.cancel()
        isRecording.set(false)
        audioRecord?.release()
        audioRecord = null
    }

    // --- WAV Header Utilities ---

    private fun writeWavHeader(out: java.io.OutputStream, totalAudioLen: Long, totalDataLen: Long, sampleRate: Int, channels: Int, byteRate: Int) {
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
        header[20] = 1; header[21] = 0
        header[22] = channels.toByte(); header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (channels * 16 / 8).toByte(); header[33] = 0
        header[34] = 16; header[35] = 0
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
        out.write(header, 0, 44)
    }

    private fun updateWavHeader(file: File, totalAudioLen: Long) {
        val totalDataLen = totalAudioLen + 36
        val channels = 1
        val byteRate = sampleRate * 2 * channels
        
        try {
            val randomAccessFile = RandomAccessFile(file, "rw")
            randomAccessFile.seek(0)
            
            // Re-generate header
            val header = ByteArray(44)
            header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
            header[4] = (totalDataLen and 0xff).toByte()
            header[5] = ((totalDataLen shr 8) and 0xff).toByte()
            header[6] = ((totalDataLen shr 16) and 0xff).toByte()
            header[7] = ((totalDataLen shr 24) and 0xff).toByte()
            header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
            header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
            header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
            header[20] = 1; header[21] = 0
            header[22] = channels.toByte(); header[23] = 0
            header[24] = (sampleRate and 0xff).toByte()
            header[25] = ((sampleRate shr 8) and 0xff).toByte()
            header[26] = ((sampleRate shr 16) and 0xff).toByte()
            header[27] = ((sampleRate shr 24) and 0xff).toByte()
            header[28] = (byteRate and 0xff).toByte()
            header[29] = ((byteRate shr 8) and 0xff).toByte()
            header[30] = ((byteRate shr 16) and 0xff).toByte()
            header[31] = ((byteRate shr 24) and 0xff).toByte()
            header[32] = (channels * 16 / 8).toByte(); header[33] = 0
            header[34] = 16; header[35] = 0
            header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
            header[40] = (totalAudioLen and 0xff).toByte()
            header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
            header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
            header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
            
            randomAccessFile.write(header, 0, 44)
            randomAccessFile.close()
        } catch (e: Exception) {
            Log.e("Enrollment", "Error updating wav header", e)
        }
    }
}
