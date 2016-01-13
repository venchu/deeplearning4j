package org.deeplearning4j.models.sequencevectors.classes;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.math.BigDecimal;
import java.util.Calendar;
import java.util.Date;

/**
 * This class represents single Transaction made my abstract account holder.
 * It has no real meaning, and can't be used for real-word transaction identification,
 * since it only represents simplified difference between two consequent transactions time and geographical distance.
 *
 *
 * @author raver119@gmail.com
 */
public class Transaction extends SequenceElement {

    // account id
    @Getter @Setter private long accountId;

    // time when transaction happened
    @Getter @Setter private long timeMilliseconds;

    // time of the day, where transaction happened. we want to distinguish morning/day/evening/night transactions
    @Getter @Setter private int timeOfDay;

    // coordinates where transaction happened
    @Getter @Setter private BigDecimal longitude;
    @Getter @Setter private BigDecimal latitude;

    // ID of POS terminal where transaction was made
    @Getter @Setter private int posTerminalId;

    // id of the good purchased
    @Getter @Setter private int goodId;

    // type of this transaction
    @Getter @Setter private int transactionType;

    // sum of this transaction
    @Getter @Setter private BigDecimal transactionSum;

    private String label;

    /**
     * This construction is used for sequence labelling only
     * @param accountId
     */
    public Transaction(long accountId) {
        this.accountId = accountId;
        this.label = new String(Long.valueOf(accountId).toString());
    }

    /**
     * Constructor takes POS terminal number, and converts it into geo coordinates internally
     *
     * @param posTerminalNumber
     */
    public Transaction(int posTerminalNumber, long timeMilDate, int goodId, int transactionType, BigDecimal transactionSum) {
        this.posTerminalId = posTerminalNumber;
        this.timeMilliseconds = timeMilDate;
        this.transactionType = transactionType;
        this.transactionSum = transactionSum;
        this.goodId = goodId;

        Calendar cal = Calendar.getInstance();
        cal.setTimeInMillis(GeoConv.mildateToMilliseconds(timeMilDate));
        int hours = cal.get(Calendar.HOUR_OF_DAY);

        timeOfDay = hours / 8;
    }

    /**
     * In this example we don't care about real meaning of the label, since we're learning sequences.
     * So all we care about is unique labels, nothing else.
     *
     * We'll provide uniqueness based on:
     * 1. transaction type
     * 2. pos terminal id
     * 3. time of the day
     * 4. good id
     *
     * @return
     */
    @Override
    public String getLabel() {
        if (label == null) {
            StringBuilder builder = new StringBuilder();
            builder.append(transactionType).append(" ");
            builder.append(posTerminalId).append(" ");
            builder.append(timeOfDay).append(" ");
            builder.append(goodId);

            label = builder.toString();
        }
        return label;
    }

    /**
     * @return
     */
    @Override
    public String toJSON() {
        return null;
    }


}
