package com.example.driverguardian.ui.history

import com.example.driverguardian.domain.model.DrowsinessEvent
import org.junit.Assert.assertEquals
import org.junit.Test

class TripPresentationTest {
    @Test fun `duration is rendered from server seconds`() {
        assertEquals("01:01:01", formatDuration(3661))
    }

    @Test fun `nullable values use em dash and are not fabricated`() {
        assertEquals("—", formatNullableNumber(null))
        assertEquals("—", formatDateTime(null))
    }

    @Test fun `event counts come from persisted warning and danger events`() {
        val events = listOf(event(1, "WARNING", 1), event(2, "WARNING", 1), event(3, "DANGER", 2))
        assertEquals(EventCounts(warnings = 2, dangers = 1), countEvents(events))
    }

    private fun event(id: Int, state: String, level: Int) = DrowsinessEvent(
        id, 4, "2026-09-28T00:0$id:00", state, level, null, null, "N", "SYNCED"
    )
}
