package com.voice.shield

import android.app.AlertDialog
import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatDelegate
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.google.android.material.switchmaterial.SwitchMaterial

class FragmentSettings : Fragment(R.layout.fragment_settings) {

    private lateinit var prefs: SharedPreferences
    private lateinit var authPrefs: SharedPreferences

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        prefs = requireContext().getSharedPreferences("TrustVoicePrefs", Context.MODE_PRIVATE)
        authPrefs = requireContext().getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)

        // Find UI elements
        val llUserSection = view.findViewById<LinearLayout>(R.id.ll_user_section)
        val tvProfileName = view.findViewById<TextView>(R.id.tv_profile_name)
        val tvProfileEmail = view.findViewById<TextView>(R.id.tv_profile_email)
        val btnEditProfile = view.findViewById<TextView>(R.id.btn_edit_profile)
        val llProfileInfo = view.findViewById<LinearLayout>(R.id.ll_profile_info)
        val llChangePassword = view.findViewById<LinearLayout>(R.id.ll_change_password)
        val tvPreferencesHeader = view.findViewById<TextView>(R.id.tv_preferences_header)
        val switchDarkMode = view.findViewById<SwitchMaterial>(R.id.switch_dark_mode)
        val btnLogout = view.findViewById<Button>(R.id.btn_logout)
        val btnLoginOption = view.findViewById<Button>(R.id.btn_login_option)

        fun updateUiForAuthState(isLoggedIn: Boolean) {
            if (isLoggedIn) {
                llUserSection.visibility = View.VISIBLE
                tvPreferencesHeader.visibility = View.VISIBLE
                btnLogout.visibility = View.VISIBLE
                btnLoginOption.visibility = View.GONE

                val currentName = prefs.getString("profile_name", "Ganesh") ?: "Ganesh"
                val currentEmail = prefs.getString("profile_email", "ganesh@example.com") ?: "ganesh@example.com"
                tvProfileName.text = currentName
                tvProfileEmail.text = currentEmail
            } else {
                // When logged out: NO information of user should display over there
                // Only dark mode toggle should be there and login option
                llUserSection.visibility = View.GONE
                tvPreferencesHeader.visibility = View.GONE
                btnLogout.visibility = View.GONE
                btnLoginOption.visibility = View.VISIBLE
            }
        }

        // Check if currently authenticated
        val isLoggedIn = authPrefs.getString("jwt_token", null) != null
        updateUiForAuthState(isLoggedIn)

        // Load dark mode state
        val isDarkMode = prefs.getBoolean("dark_mode", false)
        switchDarkMode.isChecked = isDarkMode

        // Edit Profile & Profile Info Actions
        val editProfileListener = View.OnClickListener {
            showEditProfileDialog(tvProfileName, tvProfileEmail)
        }
        btnEditProfile.setOnClickListener(editProfileListener)
        llProfileInfo.setOnClickListener(editProfileListener)

        // Change Password Action
        llChangePassword.setOnClickListener {
            showChangePasswordDialog()
        }

        // Dark Mode Action
        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("dark_mode", isChecked).apply()
            if (isChecked) {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
            } else {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
            }
        }

        // Logout Action
        btnLogout.setOnClickListener {
            // Clear user session completely
            authPrefs.edit().remove("jwt_token").apply()
            prefs.edit().apply {
                remove("profile_name")
                remove("profile_email")
                apply()
            }

            // Immediately switch UI state: hide all user info, show only dark mode & login option
            updateUiForAuthState(false)

            Toast.makeText(requireContext(), "Logged out successfully", Toast.LENGTH_SHORT).show()
        }

        // Login Option Action (navigates to Login page)
        btnLoginOption.setOnClickListener {
            findNavController().navigate(R.id.action_settings_to_login)
        }
    }

    private fun showEditProfileDialog(tvName: TextView, tvEmail: TextView) {
        val context = requireContext()
        val layout = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 40, 50, 10)
        }

        val nameInput = EditText(context).apply {
            hint = "Full Name"
            setText(prefs.getString("profile_name", "Ganesh"))
        }
        val emailInput = EditText(context).apply {
            hint = "Email Address"
            setText(prefs.getString("profile_email", "ganesh@example.com"))
        }

        layout.addView(nameInput)
        layout.addView(emailInput)

        AlertDialog.Builder(context)
            .setTitle("Edit Profile")
            .setView(layout)
            .setPositiveButton("Save") { _, _ ->
                val newName = nameInput.text.toString().trim()
                val newEmail = emailInput.text.toString().trim()

                if (newName.isNotEmpty() && newEmail.isNotEmpty()) {
                    prefs.edit().putString("profile_name", newName)
                        .putString("profile_email", newEmail)
                        .apply()
                    tvName.text = newName
                    tvEmail.text = newEmail
                    Toast.makeText(context, "Profile updated", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(context, "Fields cannot be empty", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showChangePasswordDialog() {
        val context = requireContext()
        val layout = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 40, 50, 10)
        }

        val oldPassInput = EditText(context).apply {
            hint = "Old Password"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val newPassInput = EditText(context).apply {
            hint = "New Password"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val confirmPassInput = EditText(context).apply {
            hint = "Confirm New Password"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }

        layout.addView(oldPassInput)
        layout.addView(newPassInput)
        layout.addView(confirmPassInput)

        AlertDialog.Builder(context)
            .setTitle("Change Password")
            .setView(layout)
            .setPositiveButton("Update") { _, _ ->
                val oldP = oldPassInput.text.toString()
                val newP = newPassInput.text.toString()
                val confP = confirmPassInput.text.toString()

                if (oldP.isEmpty() || newP.isEmpty() || confP.isEmpty()) {
                    Toast.makeText(context, "Please fill all fields", Toast.LENGTH_SHORT).show()
                } else if (newP != confP) {
                    Toast.makeText(context, "New passwords do not match", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(context, "Password updated successfully", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
