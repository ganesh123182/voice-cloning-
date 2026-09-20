package com.voice.shield

import android.content.Context
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.voice.shield.api.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class FragmentSignup : Fragment(R.layout.fragment_signup) {

    private var isPasswordVisible = false

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val authPrefs = requireActivity().getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)
        val tvPrefs = requireActivity().getSharedPreferences("TrustVoicePrefs", Context.MODE_PRIVATE)

        val btnBack = view.findViewById<ImageView>(R.id.btn_back)
        val etFullname = view.findViewById<EditText>(R.id.et_fullname)
        val etEmail = view.findViewById<EditText>(R.id.et_email)
        val etPhone = view.findViewById<EditText>(R.id.et_phone)
        val etCreatePassword = view.findViewById<EditText>(R.id.et_create_password)
        val ivToggleSignupPassword = view.findViewById<ImageView>(R.id.iv_toggle_signup_password)
        val cbTerms = view.findViewById<CheckBox>(R.id.cb_terms)
        val btnSignup = view.findViewById<Button>(R.id.btn_signup)
        val progressSignup = view.findViewById<ProgressBar>(R.id.progress_signup)
        val btnToLogin = view.findViewById<TextView>(R.id.btn_to_login)

        // Back button
        btnBack.setOnClickListener {
            findNavController().navigateUp()
        }

        // Navigate to Login
        btnToLogin.setOnClickListener {
            findNavController().navigateUp()
        }

        // Toggle password visibility
        ivToggleSignupPassword.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            if (isPasswordVisible) {
                etCreatePassword.inputType = InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
            } else {
                etCreatePassword.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            }
            etCreatePassword.setSelection(etCreatePassword.text.length)
        }

        // Sign Up button
        btnSignup.setOnClickListener {
            val fullName = etFullname.text.toString().trim()
            val email = etEmail.text.toString().trim()
            val phone = etPhone.text.toString().trim()
            val password = etCreatePassword.text.toString().trim()

            if (fullName.isEmpty()) {
                etFullname.error = "Please enter your Full Name"
                etFullname.requestFocus()
                return@setOnClickListener
            }
            if (email.isEmpty() && phone.isEmpty()) {
                etEmail.error = "Please enter an Email or Phone Number"
                etEmail.requestFocus()
                return@setOnClickListener
            }
            if (password.length < 4) {
                etCreatePassword.error = "Password must be at least 4 characters"
                etCreatePassword.requestFocus()
                return@setOnClickListener
            }
            if (!cbTerms.isChecked) {
                Toast.makeText(requireContext(), "Please accept the Terms & Conditions", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val username = if (email.isNotEmpty()) email else phone

            btnSignup.text = ""
            progressSignup.visibility = View.VISIBLE
            btnSignup.isEnabled = false

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val response = RetrofitClient.instance.register(
                        u = username,
                        p = password,
                        fn = fullName,
                        em = email.ifEmpty { null },
                        ph = phone.ifEmpty { null }
                    )

                    withContext(Dispatchers.Main) {
                        progressSignup.visibility = View.GONE
                        btnSignup.text = "Sign Up"
                        btnSignup.isEnabled = true

                        val regBody = response.body()
                        val token = regBody?.access_token ?: ("demo_token_" + System.currentTimeMillis())
                        val uid = regBody?.user_id ?: ("user_" + (username.hashCode().toLong() and 0xffffffffL))
                        authPrefs.edit()
                            .putString("jwt_token", token)
                            .putString("user_id", uid)
                            .apply()
                        tvPrefs.edit()
                            .putString("profile_name", fullName)
                            .putString("profile_email", if (email.isNotEmpty()) email else phone)
                            .putString("local_user_pass_" + username, password)
                            .apply()

                        Toast.makeText(requireContext(), "Account created! Welcome, $fullName!", Toast.LENGTH_SHORT).show()
                        findNavController().navigate(R.id.action_signup_to_home)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        progressSignup.visibility = View.GONE
                        btnSignup.text = "Sign Up"
                        btnSignup.isEnabled = true

                        // Local demo registration
                        val uid = "user_" + (username.hashCode().toLong() and 0xffffffffL)
                        authPrefs.edit()
                            .putString("jwt_token", "local_demo_token_" + uid)
                            .putString("user_id", uid)
                            .apply()
                        tvPrefs.edit()
                            .putString("profile_name", fullName)
                            .putString("profile_email", if (email.isNotEmpty()) email else phone)
                            .putString("local_user_pass_" + username, password)
                            .apply()

                        Toast.makeText(requireContext(), "Account created! Welcome, $fullName!", Toast.LENGTH_SHORT).show()
                        findNavController().navigate(R.id.action_signup_to_home)
                    }
                }
            }
        }
    }
}
