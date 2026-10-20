import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Đọc file .jsonl của Amazon Reviews'23, giữ 4 cột rating, text, asin, user_id,
 * lấy mẫu ngẫu nhiên trên toàn file (reservoir sampling) rồi chia thành 2 file ARFF train/test.
 */
class HealthCareArff {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** @param maxTrain số dòng train; test lấy thêm theo tỉ lệ trainRatio. maxTrain <= 0 = lấy hết. */
    public static void convert(Path input, Path trainOut, Path testOut,
                               int maxTrain, double trainRatio, long seed, String relation) throws IOException {
        int sampleSize = maxTrain > 0 ? (int) Math.round(maxTrain / trainRatio) : Integer.MAX_VALUE;
        Random rnd = new Random(seed);
        List<String> rows = new ArrayList<>();
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

                String row;
                try {
                    row = toRow(MAPPER.readTree(line));
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
        write(trainOut, relation + "-train", rows.subList(0, trainSize));
        write(testOut, relation + "-test", rows.subList(trainSize, rows.size()));

        System.out.printf("Doc %,d dong, lay mau %,d%n", seen, rows.size());
        System.out.printf("Train: %,d dong -> %s%n", trainSize, trainOut);
        System.out.printf("Test : %,d dong -> %s%n", rows.size() - trainSize, testOut);
        if (skipped > 0) System.out.printf("Bo qua %,d dong JSON loi%n", skipped);
    }

    private static void write(Path out, String relation, List<String> rows) throws IOException {
        Files.createDirectories(out.toAbsolutePath().getParent());
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("@relation " + Utils.quote(relation) + "\n\n");
            w.write("@attribute rating {1,2,3,4,5}\n");
            w.write("@attribute text string\n");
            w.write("@attribute asin string\n");
            w.write("@attribute user_id string\n\n");
            w.write("@data\n");
            for (String row : rows) w.write(row);
        }
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
                20_000, // số dòng train (test tự lấy thêm theo tỉ lệ -> 5.000)
                0.8,    // 80% train / 20% test
                42,     // seed cố định để lần nào chạy cũng ra cùng kết quả
                name);
    }
}
