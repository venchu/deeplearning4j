package org.deeplearning4j.models.sequencevectors.classes;

import org.junit.Before;
import org.junit.Test;

import java.util.Calendar;
import java.util.Date;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class GeoConvTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testMildateToMilliseconds() throws Exception {
        long millis = GeoConv.mildateToMilliseconds(20150504200515L);

      //  Date date = new Date(millis);
        Calendar calendar = Calendar.getInstance();
        calendar.setTimeInMillis(millis);
        assertEquals(20, calendar.get(Calendar.HOUR_OF_DAY));
        assertEquals(2015, calendar.get(Calendar.YEAR));
        assertEquals(5, calendar.get(Calendar.MONTH));
        assertEquals(4, calendar.get(Calendar.DAY_OF_MONTH));
        assertEquals(5, calendar.get(Calendar.MINUTE));
        assertEquals(15, calendar.get(Calendar.SECOND));
    }
}