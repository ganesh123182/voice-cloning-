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
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
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
    
    private var audioRecord: AudioRecord? = null
    private var audioFile: File? = null
    private val isRecording = java.util.concurrent.atomic.AtomicBoolean(false)
    private var recordingJob: Job? = null
    
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

        btnStartRecording.setOnClickListener {
            if (isRecording.get()) {
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
        
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, minBufSize * 2)
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Toast.makeText(requireContext(), "Failed to initialize microphone", Toast.LENGTH_SHORT).show()
            isRecording.set(false)
            resetButton()
            return
        }

        audioFile = File(requireContext().cacheDir, "enrollment_audio.wav")
        audioRecord?.startRecording()

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val out = FileOutputStream(audioFile)
                writeWavHeader(out, 0, 0, sampleRate, 1, sampleRate * 2) // Dummy header
                
                val buffer = ByteArray(minBufSize * 2)
                var totalAudioLen = 0L
                while (isRecording.get()) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        out.write(buffer, 0, read)
                        totalAudioLen += read
                    }
                }
                out.close()
                
                // Overwrite the header with accurate file lengths
                updateWavHeader(audioFile!!, totalAudioLen)
            } catch (e: Exception) {
                Log.e("Enrollment", "Error writing audio file", e)
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
        recordingJob?.cancel()
        resetButton()
        isRecording.set(false)

        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.e("Enrollment", "Error stopping recorder", e)
        } finally {
            audioRecord?.release()
            audioRecord = null
        }

        // Delay slightly to ensure IO thread finishes writing file
        lifecycleScope.launch {
            delay(200)
            uploadAudio()
        }
    }
    
    private fun resetButton() {
        btnStartRecording.text = getString(R.string.start_recording)
        btnStartRecording.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.trustvoice_primary))
    }

    private fun uploadAudio() {
        if (audioFile == null || !audioFile!!.exists()) return
        if (audioFile!!.length() < 100L) {
            Toast.makeText(requireContext(), "Recording was too short. Please try again.", Toast.LENGTH_SHORT).show()
            return
        }

        progressEnrollment.visibility = View.VISIBLE
        btnStartRecording.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Send as audio/wav since we generated a proper RIFF WAV file
                val prefs = requireActivity().getSharedPreferences("AuthPrefs", android.content.Context.MODE_PRIVATE)
                val token = prefs.getString("jwt_token", null)
                if (token == null) {
                    withContext(Dispatchers.Main) { Toast.makeText(requireContext(), "Not Authenticated", Toast.LENGTH_SHORT).show() }
                    return@launch
                }
                
                val requestFile = audioFile!!.asRequestBody("audio/wav".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", audioFile!!.name, requestFile)
                
                val response = RetrofitClient.instance.enrollVoice("Bearer $token", body)

                withContext(Dispatchers.Main) {
                    progressEnrollment.visibility = View.GONE
                    btnStartRecording.isEnabled = true
                    
                    if (response.isSuccessful) {
                        val respBody = response.body()
                        val hash = respBody?.evidence_hash?.take(8) ?: ""
                        val tx = respBody?.blockchain_tx_hash?.take(8) ?: "N/A"
                        Toast.makeText(requireContext(), "Enrolled! ID: ${respBody?.enrollment_id}\nHash: $hash\nTx: $tx", Toast.LENGTH_LONG).show()
                    } else {
                        Toast.makeText(requireContext(), "Enrollment failed: ${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressEnrollment.visibility = View.GONE
                    btnStartRecording.isEnabled = true
                    Toast.makeText(requireContext(), "Network Error", Toast.LENGTH_SHORT).show()
                }
                Log.e("Enrollment", "Error uploading", e)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
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
