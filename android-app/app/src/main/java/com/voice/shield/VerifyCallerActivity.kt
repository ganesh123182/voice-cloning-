package com.voice.shield

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class VerifyCallerActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_verify_caller)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars =
                insets.getInsets(WindowInsetsCompat.Type.systemBars())

            v.setPadding(
                systemBars.left,
                systemBars.top,
                systemBars.right,
                systemBars.bottom
            )

            insets
        }

        // Verification completed
        findViewById<Button>(R.id.btn_confirm).setOnClickListener {
            val intent = Intent(this, CallDetailsActivity::class.java)
            startActivity(intent)
            finish()
        }

        // Back to previous screen
        findViewById<Button>(R.id.btn_cancel).setOnClickListener {
            finish()
        }
    }
}