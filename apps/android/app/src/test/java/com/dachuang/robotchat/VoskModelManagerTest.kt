package com.dachuang.robotchat

import com.dachuang.robotchat.speech.VoskModelManager
import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoskModelManagerTest {
    @Test
    fun acceptsOfficialSmallChineseModelStructureWithoutUuid() {
        withTempModel { model ->
            requiredModelFiles.forEach { model.createNonEmptyFile(it) }

            assertFalse(File(model, "uuid").exists())
            assertTrue(VoskModelManager.isModelComplete(model))
        }
    }

    @Test
    fun rejectsModelWithMissingOrEmptyRequiredFile() {
        withTempModel { model ->
            requiredModelFiles.forEach { model.createNonEmptyFile(it) }
            File(model, "graph/HCLr.fst").writeBytes(byteArrayOf())

            assertFalse(VoskModelManager.isModelComplete(model))
        }
    }

    private fun withTempModel(block: (File) -> Unit) {
        val directory = Files.createTempDirectory("vosk-model-test").toFile()
        try {
            block(directory)
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun File.createNonEmptyFile(relativePath: String) {
        File(this, relativePath).apply {
            parentFile?.mkdirs()
            writeText("test")
        }
    }

    private companion object {
        val requiredModelFiles = listOf(
            "am/final.mdl",
            "conf/mfcc.conf",
            "conf/model.conf",
            "graph/Gr.fst",
            "graph/HCLr.fst",
            "graph/phones/word_boundary.int",
            "ivector/final.ie",
        )
    }
}
