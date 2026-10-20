import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Đọc file .jsonl của Amazon Reviews'23, giữ 4 cột rating, text, asin, user_id
 * rồi chia ngẫu nhiên thành 2 file ARFF train/test (đọc-ghi từng dòng, không load cả file vào RAM).
 */
class HealthCareArff {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void convert(Path input, Path trainOut, Path testOut,
                               double trainRatio, long seed, String relation) throws IOException {
        Files.createDirectories(trainOut.toAbsolutePath().getParent());
        Random rnd = new Random(seed);
        int train = 0, test = 0, skipped = 0;

        try (BufferedReader in = Files.newBufferedReader(input, StandardCharsets.UTF_8);
             BufferedWriter trainW = Files.newBufferedWriter(trainOut, StandardCharsets.UTF_8);
             BufferedWriter testW = Files.newBufferedWriter(testOut, StandardCharsets.UTF_8)) {

            writeHeader(trainW, relation + "-train");
            writeHeader(testW, relation + "-test");

            String line;
            while ((line = in.readLine()) != null) {
                if (line.isBlank()) continue;
                JsonNode node;
                try {
                    node = MAPPER.readTree(line);
                } catch (IOException e) {
                    skipped++;
                    continue;
                }

                String row = toRow(node);
                if (rnd.nextDouble() < trainRatio) {
                    trainW.write(row);
                    train++;
                } else {
                    testW.write(row);
                    test++;
                }
            }
        }

        System.out.printf("Train: %,d dong -> %s%n", train, trainOut);
        System.out.printf("Test : %,d dong -> %s%n", test, testOut);
        if (skipped > 0) System.out.printf("Bo qua %,d dong JSON loi%n", skipped);
    }

    private static void writeHeader(BufferedWriter w, String relation) throws IOException {
        w.write("@relation " + Utils.quote(relation) + "\n\n");
        w.write("@attribute rating {1,2,3,4,5}\n");
        w.write("@attribute text string\n");
        w.write("@attribute asin string\n");
        w.write("@attribute user_id string\n\n");
        w.write("@data\n");
    }

    private static String toRow(JsonNode n) {
        JsonNode r = n.get("rating");
        String rating = (r == null || r.isNull()) ? "?" : String.valueOf((int) Math.round(r.asDouble()));
        return rating + ","
                + str(n, "text") + ","
                + str(n, "asin") + ","
                + str(n, "user_id") + "\n";
    }

    /** Giá trị string trong ARFF: thiếu -> "?", còn lại quote + escape (dấu ', xuống dòng, ...) theo chuẩn Weka. */
    private static String str(JsonNode n, String field) {
        JsonNode v = n.get(field);
        return (v == null || v.isNull()) ? "?" : Utils.quote(v.asText());
    }
}

public class HealthCareArffTest {

    public static void main(String[] args) throws IOException {
        String name = "Health_and_Personal_Care";
        HealthCareArff.convert(
                Path.of("raw-data", name + ".jsonl"),
                Path.of("datasets", name + "-train.arff"),
                Path.of("datasets", name + "-test.arff"),
                0.8,   // 80% train / 20% test
                42,    // seed cố định để lần nào chạy cũng chia giống nhau
                name);
    }
}
