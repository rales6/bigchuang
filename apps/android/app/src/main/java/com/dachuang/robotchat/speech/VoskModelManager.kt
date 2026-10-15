package com.dachuang.robotchat.speech

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

object VoskModelManager {
    const val MODEL_NAME = "vosk-model-small-cn-0.22"
    const val MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"

    private val requiredFiles = listOf(
        "am/final.mdl",
        "conf/mfcc.conf",
        "conf/model.conf",
        "graph/Gr.fst",
        "graph/HCLr.fst",
        "graph/phones/word_boundary.int",
        "ivector/final.ie",
    )

    fun modelDir(context: Context): File =
        context.getExternalFilesDir("vosk") ?: File(context.filesDir, "vosk")

    fun modelPath(context: Context): File = File(modelDir(context), MODEL_NAME)

    fun isReady(context: Context): Boolean = isModelComplete(modelPath(context))

    /**
     * Validates the files Vosk actually loads. The official small Chinese model does not
     * contain a `uuid` file, so using that as an installation marker rejects a valid model.
     */
    fun isModelComplete(path: File): Boolean =
        path.isDirectory && requiredFiles.all { relativePath ->
            File(path, relativePath).let { it.isFile && it.length() > 0L }
        }

    fun installedSizeBytes(context: Context): Long {
        val path = modelPath(context)
        if (!path.exists()) return 0L
        return path.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }

    fun deleteModel(context: Context): Boolean {
        val dir = modelDir(context)
        val paths = listOf(
            modelPath(context),
            File(dir, "$MODEL_NAME.installing"),
            File(dir, "$MODEL_NAME.backup"),
            File(dir, "$MODEL_NAME.zip.tmp"),
        )
        return paths.all { !it.exists() || it.deleteRecursively() }
    }

    suspend fun download(
        context: Context,
        onProgress: (Float) -> Unit,
    ): File = withContext(Dispatchers.IO) {
        val dir = modelDir(context)
        if (!dir.exists() && !dir.mkdirs()) {
            throw RuntimeException("无法创建模型存储目录")
        }
        val zipFile = File(dir, "$MODEL_NAME.zip.tmp")
        val stagingDir = File(dir, "$MODEL_NAME.installing")
        val backupDir = File(dir, "$MODEL_NAME.backup")
        var connection: HttpURLConnection? = null

        try {
            zipFile.delete()
            stagingDir.deleteRecursively()
            backupDir.deleteRecursively()

            connection = URL(MODEL_URL).openConnection() as HttpURLConnection
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("Accept-Encoding", "identity")
            connection.connectTimeout = 15_000
            connection.readTimeout = 60_000
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw RuntimeException("服务器返回 ${connection.responseCode}")
            }

            val contentLength = connection.contentLengthLong
            connection.inputStream.use { input ->
                FileOutputStream(zipFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead = 0L
                    var length: Int
                    while (input.read(buffer).also { length = it } != -1) {
                        output.write(buffer, 0, length)
                        bytesRead += length
                        if (contentLength > 0L) {
                            onProgress(bytesRead.toFloat() / contentLength.toFloat())
                        } else {
                            onProgress(-1f)
                        }
                    }
                }
            }
            if (contentLength > 0L && zipFile.length() != contentLength) {
                throw RuntimeException("模型下载不完整，请检查网络后重试")
            }

            onProgress(-1f)
            if (!stagingDir.mkdirs()) {
                throw RuntimeException("无法创建模型安装目录")
            }
            val stagingRoot = stagingDir.canonicalFile
            val stagingPrefix = stagingRoot.path + File.separator

            ZipInputStream(zipFile.inputStream().buffered()).use { zipInput ->
                var entry = zipInput.nextEntry
                while (entry != null) {
                    val targetFile = File(stagingRoot, entry.name).canonicalFile
                    if (targetFile.path != stagingRoot.path &&
                        !targetFile.path.startsWith(stagingPrefix)
                    ) {
                        throw RuntimeException("模型压缩包包含非法路径")
                    }
                    if (entry.isDirectory) {
                        if (!targetFile.exists() && !targetFile.mkdirs()) {
                            throw RuntimeException("无法创建模型目录")
                        }
                    } else {
                        targetFile.parentFile?.mkdirs()
                        FileOutputStream(targetFile).use { output ->
                            zipInput.copyTo(output)
                        }
                    }
                    zipInput.closeEntry()
                    entry = zipInput.nextEntry
                }
            }

            val extractedModel = when {
                isModelComplete(stagingDir) -> stagingDir
                else -> stagingDir.listFiles()
                    ?.firstOrNull { it.isDirectory && isModelComplete(it) }
            } ?: throw RuntimeException("模型压缩包缺少语音识别所需文件")

            val finalPath = modelPath(context)
            if (finalPath.exists() && !finalPath.renameTo(backupDir)) {
                throw RuntimeException("无法替换旧模型，请检查存储空间")
            }
            if (!extractedModel.renameTo(finalPath)) {
                if (backupDir.exists()) backupDir.renameTo(finalPath)
                throw RuntimeException("模型安装失败，请检查存储空间")
            }
            backupDir.deleteRecursively()

            finalPath
        } finally {
            connection?.disconnect()
            zipFile.delete()
            stagingDir.deleteRecursively()
            if (backupDir.exists() && !modelPath(context).exists()) {
                backupDir.renameTo(modelPath(context))
            }
        }
    }

    fun formattedSize(context: Context): String {
        val bytes = installedSizeBytes(context)
        if (bytes <= 0) return "未安装"
        val mb = bytes / (1024.0 * 1024.0)
        return "%.1f MB".format(mb)
    }
}
