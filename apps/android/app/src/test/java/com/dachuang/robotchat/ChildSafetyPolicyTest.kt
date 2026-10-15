package com.dachuang.robotchat

import com.dachuang.robotchat.data.ChildSafetyPolicy
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChildSafetyPolicyTest {
    @Test
    fun allowsConversationAndEmergencyStop() {
        assertTrue(ChildSafetyPolicy.validate("给我讲一个故事").allowed)
        assertTrue(ChildSafetyPolicy.validate("请立即停止").allowed)
    }

    @Test
    fun blocksMotionCommandsAndLongText() {
        assertFalse(ChildSafetyPolicy.validate("快速前进并抓取杯子").allowed)
        assertFalse(ChildSafetyPolicy.validate("好".repeat(81)).allowed)
    }
}
