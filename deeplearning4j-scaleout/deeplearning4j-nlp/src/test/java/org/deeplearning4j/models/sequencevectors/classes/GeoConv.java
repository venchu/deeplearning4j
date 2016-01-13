package org.deeplearning4j.models.sequencevectors.classes;

import java.util.Date;

/**
 * Static utils needed for SequenceVectors transactions representation test
 * @author raver119@gmail.com
 */
public class GeoConv {

    public static long mildateToMilliseconds(long mildate) {
        String string = Long.valueOf(mildate).toString();

        int year = Integer.valueOf(string.substring(0,4));
        int month = Integer.valueOf(string.substring(4,6));
        int day = Integer.valueOf(string.substring(6,8));
        int hour24 = Integer.valueOf(string.substring(8,10));
        int minute = Integer.valueOf(string.substring(10,12));
        int seconds = Integer.valueOf(string.substring(12,14));

        Date date = new Date();
        date.setYear(year - 1900);
        date.setMonth(month);
        date.setDate(day);
        date.setHours(hour24);
        date.setMinutes(minute);
        date.setSeconds(seconds);
        return date.getTime();
    }
}
