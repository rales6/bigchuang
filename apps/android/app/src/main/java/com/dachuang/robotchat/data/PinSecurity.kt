package com.dachuang.robotchat.data

import android.content.SharedPreferences
import android.util.Base64
import androidx.core.content.edit
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

class PinSecurity(private val preferences: SharedPreferences) {
    private val saltKey = "guardian_pin_salt"
    private val hashKey = "guardian_pin_hash"

    fun hasPin(): Boolean = preferences.contains(saltKey) && preferences.contains(hashKey)

    fun setPin(pin: String): Boolean {
        if (!pin.matches(Regex("\\d{6}"))) return false
        val salt = ByteArray(16).also(SecureRandom()::nextBytes)
        val hash = derive(pin, salt)
        preferences.edit {
            putString(saltKey, Base64.encodeToString(salt, Base64.NO_WRAP))
            putString(hashKey, Base64.encodeToString(hash, Base64.NO_WRAP))
        }
        return true
    }

    fun verify(pin: String): Boolean {
        val salt = preferences.getString(saltKey, null)?.decodeBase64() ?: return false
        val expected = preferences.getString(hashKey, null)?.decodeBase64() ?: return false
        return MessageDigest.isEqual(expected, derive(pin, salt))
    }

    private fun derive(pin: String, salt: ByteArray): ByteArray {
        val spec = PBEKeySpec(pin.toCharArray(), salt, 120_000, 256)
        return try {
            SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded
        } finally {
            spec.clearPassword()
        }
    }

    private fun String.decodeBase64(): ByteArray? = runCatching {
        Base64.decode(this, Base64.NO_WRAP)
    }.getOrNull()
}
