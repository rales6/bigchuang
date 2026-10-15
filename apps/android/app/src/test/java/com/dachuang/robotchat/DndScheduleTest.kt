package com.dachuang.robotchat

import com.dachuang.robotchat.model.AppSettings
import com.dachuang.robotchat.model.DndSchedule
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalTime

class DndScheduleTest {
    @Test
    fun overnightScheduleSpansMidnight() {
        val settings = AppSettings(
            dndScheduled = true,
            dndStartMinutes = 22 * 60,
            dndEndMinutes = 7 * 60,
        )
        assertTrue(DndSchedule.isActive(settings, LocalTime.of(23, 30)))
        assertTrue(DndSchedule.isActive(settings, LocalTime.of(6, 59)))
        assertFalse(DndSchedule.isActive(settings, LocalTime.of(12, 0)))
    }

    @Test
    fun alwaysDndOverridesSchedule() {
        assertTrue(DndSchedule.isActive(AppSettings(dndAlways = true), LocalTime.NOON))
    }
}
