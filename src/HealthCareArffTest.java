import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Đọc file .jsonl của Amazon Reviews'23, lấy mẫu ngẫu nhiên trên toàn file (reservoir sampling), chia train/test rồi:
 * - ARFF (datasets/): text đã qua StringToWordVector + rating ở cột cuối (class), mở Weka là train được ngay
 * - metadata CSV (raw-data/): asin, user_id, text gốc, cùng thứ tự dòng với file ARFF
 */
class HealthCareArff {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    record Review(String rating, String text, String asin, String userId) {}

    /** @param maxTrain số dòng train; test lấy thêm theo tỉ lệ trainRatio. maxTrain <= 0 = lấy hết. */
    public static void convert(Path input, Path arffDir, Path metaDir, String name,
                               int maxTrain, double trainRatio, long seed) throws Exception {
        int sampleSize = maxTrain > 0 ? (int) Math.round(maxTrain / trainRatio) : Integer.MAX_VALUE;
        Random rnd = new Random(seed);
        List<Review> rows = new ArrayList<>();
        long seen = 0;
        int skipped = 0;

        // Reservoir sampling: mỗi dòng trong file có xác suất được chọn như nhau, chỉ giữ sampleSize dòng trong RAM
        try (BufferedReader in = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String line;
            while ((line = in.readLine()) != null) {
                if (line.isBlank()) continue;
                int slot = seen < sampleSize ? -1 : (int) Math.min(rnd.nextLong(seen + 1), Integer.MAX_VALUE);
                seen++;
                if (slot >= sampleSize) continue;   // không được chọn -> khỏi parse JSON

                Review row;
                try {
                    row = toReview(MAPPER.readTree(line));
                } catch (IOException e) {
                    skipped++;
                    continue;
                }
                if (slot < 0) rows.add(row);
                else rows.set(slot, row);
            }
        }

        Collections.shuffle(rows, rnd);
        int trainSize = maxTrain > 0 ? Math.min(maxTrain, rows.size()) : (int) Math.round(rows.size() * trainRatio);
        List<Review> trainRows = rows.subList(0, trainSize);
        List<Review> testRows = rows.subList(trainSize, rows.size());

        // Bộ từ vựng chỉ học từ train, test dùng lại đúng filter đó -> 2 file cùng cột
        StringToWordVector words = new StringToWordVector();
        words.setLowerCaseTokens(true);
        words.setOutputWordCounts(true);
        words.setWordsToKeep(1000);
        words.setDoNotOperateOnPerClassBasis(true);
        words.setStopwordsHandler(new Rainbow());
        words.setTokenizer(new AlphabeticTokenizer());  // chỉ lấy từ gồm chữ cái, bỏ số và ký tự lạ

        Instances train = toInstances(trainRows);
        words.setInputFormat(train);
        Instances trainVec = Filter.useFilter(train, words);
        Instances testVec = Filter.useFilter(toInstances(testRows), words);

        // Đưa rating xuống cột cuối để Weka tự nhận làm class
        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-last,1");
        reorder.setInputFormat(trainVec);
        trainVec = Filter.useFilter(trainVec, reorder);
        testVec = Filter.useFilter(testVec, reorder);

        Path trainArff = arffDir.resolve(name + "-train.arff");
        Path testArff = arffDir.resolve(name + "-test.arff");
        Path trainMeta = metaDir.resolve(name + "-train-metadata.csv");
        Path testMeta = metaDir.resolve(name + "-test-metadata.csv");
        saveArff(trainVec, name + "-train", trainArff);
        saveArff(testVec, name + "-test", testArff);
        saveMetadata(trainRows, trainMeta);
        saveMetadata(testRows, testMeta);

        System.out.printf("Doc %,d dong, lay mau %,d%n", seen, rows.size());
        System.out.printf("Train: %,d dong x %,d cot -> %s, %s%n", trainVec.numInstances(), trainVec.numAttributes(), trainArff, trainMeta);
        System.out.printf("Test : %,d dong x %,d cot -> %s, %s%n", testVec.numInstances(), testVec.numAttributes(), testArff, testMeta);
        if (skipped > 0) System.out.printf("Bo qua %,d dong JSON loi%n", skipped);
    }

    private static Review toReview(JsonNode n) {
        JsonNode r = n.get("rating");
        String rating = (r == null || r.isNull()) ? null : String.valueOf((int) Math.round(r.asDouble()));
        return new Review(rating, text(n, "text"), text(n, "asin"), text(n, "user_id"));
    }

    private static String text(JsonNode n, String field) {
        JsonNode v = n.get(field);
        return (v == null || v.isNull()) ? null : v.asText();
    }

    /** Instances 2 cột (rating, text) với class = rating, để đưa vào StringToWordVector. */
    private static Instances toInstances(List<Review> rows) {
        ArrayList<Attribute> attrs = new ArrayList<>();
        // Viết hoa để không trùng với từ "rating" (các từ sau StringToWordVector đều là chữ thường)
        attrs.add(new Attribute("Rating", List.of("1", "2", "3", "4", "5")));
        attrs.add(new Attribute("text", (List<String>) null));
        Instances data = new Instances("reviews", attrs, rows.size());
        data.setClassIndex(0);

        for (Review r : rows) {
            DenseInstance inst = new DenseInstance(2);
            inst.setDataset(data);
            if (r.rating() == null) inst.setMissing(0);
            else inst.setValue(0, r.rating());
            if (r.text() == null) inst.setMissing(1);
            else inst.setValue(1, stripHtml(r.text()));
            data.add(inst);
        }
        return data;
    }

    /** Bỏ thẻ HTML (<br />, ...) và entity (&#34;, &amp;, ...) có trong review. */
    private static String stripHtml(String s) {
        return s.replaceAll("<[^>]*>", " ").replaceAll("&#?\\w+;?", " ");
    }

    private static void saveArff(Instances data, String relation, Path out) throws IOException {
        Files.createDirectories(out.toAbsolutePath().getParent());
        data.setRelationName(relation);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(out.toFile());
        saver.writeBatch();
    }

    private static void saveMetadata(List<Review> rows, Path out) throws IOException {
        Files.createDirectories(out.toAbsolutePath().getParent());
        CSVFormat format = CSVFormat.DEFAULT.builder().setHeader("asin", "user_id", "text").build();
        try (CSVPrinter csv = new CSVPrinter(Files.newBufferedWriter(out, StandardCharsets.UTF_8), format)) {
            for (Review r : rows) csv.printRecord(r.asin(), r.userId(), r.text());
        }
    }
}

public class HealthCareArffTest {

    public static void main(String[] args) throws Exception {
        HealthCareArff.convert(
                Path.of("raw-data", "Health_and_Personal_Care.jsonl"),
                Path.of("datasets"),  // ARFF train/test
                Path.of("raw-data"),  // metadata CSV (asin, user_id, text)
                "Health_and_Personal_Care",
                20_000, // số dòng train (test tự lấy thêm theo tỉ lệ -> 5.000)
                0.8,    // 80% train / 20% test
                42);    // seed cố định để lần nào chạy cũng ra cùng kết quả
    }
}
