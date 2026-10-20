import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.stopwords.Rainbow;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Đọc file .jsonl của Amazon Reviews'23, lấy mẫu ngẫu nhiên N dòng trên toàn file (reservoir sampling)
 * rồi ghi ra 1 file ARFF (UTF-8) gồm 2 cột: text (string) và Rating (class, cột cuối).
 * Chia train/test và StringToWordVector làm trực tiếp trong Weka.
 */
public class HealthCareArffSimple {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    record Review(String text, String rating) {}

    public static void convert(Path input, Path output, int sampleSize, long seed) throws Exception {
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

        Instances data = toInstances(rows);
        Files.createDirectories(output.toAbsolutePath().getParent());
        // Ghi thẳng bằng UTF-8, không dùng ArffSaver vì nó ghi theo charset mặc định của JVM
        Files.writeString(output, data.toString(), StandardCharsets.UTF_8);

        long withVideo = rows.stream().filter(r -> r.text() != null && r.text().contains(VIDEO_TOKEN)).count();
        System.out.printf("Doc %,d dong, ghi %,d dong (%,d co video) -> %s%n", seen, data.numInstances(), withVideo, output);
        if (skipped > 0) System.out.printf("Bo qua %,d dong JSON loi%n", skipped);
    }

    private static Review toReview(JsonNode n) {
        JsonNode r = n.get("rating");
        String rating = (r == null || r.isNull()) ? null : String.valueOf((int) Math.round(r.asDouble()));
        JsonNode t = n.get("text");
        if (t == null || t.isNull()) return new Review(null, rating);

        String text = VIDEO_EMBED.matcher(t.asText()).replaceAll(" " + VIDEO_TOKEN + " ");
        text = ASIN_LINK.matcher(text).replaceAll(m -> Matcher.quoteReplacement(m.group(1)));
        return new Review(removeStopwords(stripHtml(text)), rating);
    }

    /** Instances 2 cột (text, Rating) với class = Rating ở cột cuối. */
    private static Instances toInstances(List<Review> rows) {
        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("text", (List<String>) null));
        // Viết hoa để không trùng với từ "rating" khi chạy StringToWordVector trong Weka
        attrs.add(new Attribute("Rating", List.of("1", "2", "3", "4", "5")));
        Instances data = new Instances("Health_and_Personal_Care", attrs, rows.size());
        data.setClassIndex(1);

        for (Review r : rows) {
            DenseInstance inst = new DenseInstance(2);
            inst.setDataset(data);
            if (r.text() == null) inst.setMissing(0);
            else inst.setValue(0, r.text());
            if (r.rating() == null) inst.setMissing(1);
            else inst.setValue(1, r.rating());
            data.add(inst);
        }
        return data;
    }

    // Review có video nhúng dạng [[VIDEOID:2c73ad39d6e3ee88ac30b8eb73c734f]]: dấu [ ] : và mã hex sẽ bị tokenizer
    // xé thành nhiều từ rác, nên thay bằng 1 từ liền khối -> StringToWordVector tạo đúng 1 thuộc tính "hasvideo".
    // Chữ thường, chỉ gồm chữ cái, không có trong dữ liệu gốc, không phải stopword -> giữ nguyên dù bật lowerCaseTokens.
    private static final String VIDEO_TOKEN = "hasvideo";
    private static final Pattern VIDEO_EMBED = Pattern.compile("\\[\\[VIDEOID:[^\\]]*\\]\\]");
    // Link sản phẩm dạng [[ASIN:B000LY3X7W Lewis N. Clark Sleep Mask]] -> giữ lại tên sản phẩm, bỏ mã ASIN
    private static final Pattern ASIN_LINK = Pattern.compile("\\[\\[ASIN:\\S*\\s*([^\\]]*)\\]\\]");

    private static final Pattern HTML_TAG = Pattern.compile("<[^>]*>");
    private static final Pattern HTML_ENTITY = Pattern.compile("&(#\\d+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);");
    private static final Map<String, String> NAMED_ENTITIES = Map.of(
            "quot", "\"", "apos", "'", "amp", "&", "lt", "<", "gt", ">", "nbsp", " ",
            "eacute", "é", "egrave", "è", "aacute", "á", "agrave", "à");

    /**
     * Bỏ thẻ HTML (<br />, ...) rồi giải mã entity (&#34; -> ", &amp; -> &, &eacute; -> é, ...)
     * thay vì xoá, để không làm vỡ từ. Gom nhiều khoảng trắng liền nhau thành 1.
     */
    private static String stripHtml(String s) {
        String noTags = HTML_TAG.matcher(s).replaceAll(" ");
        String decoded = HTML_ENTITY.matcher(noTags)
                .replaceAll(m -> Matcher.quoteReplacement(decodeEntity(m.group(1))));
        return decoded.replaceAll("\\s+", " ").strip();
    }

    /** body là phần giữa '&' và ';', vd "#34", "#x27", "amp". Entity lạ -> khoảng trắng. */
    private static String decodeEntity(String body) {
        try {
            if (body.startsWith("#x") || body.startsWith("#X")) return Character.toString(Integer.parseInt(body.substring(2), 16));
            if (body.startsWith("#")) return Character.toString(Integer.parseInt(body.substring(1)));
        } catch (IllegalArgumentException e) {   // số quá lớn hoặc code point không hợp lệ
            return " ";
        }
        return NAMED_ENTITIES.getOrDefault(body, " ");
    }

    // Dùng đúng danh sách Rainbow của Weka (StringToWordVector -> stopwordsHandler = Rainbow)
    private static final Rainbow STOPWORDS = new Rainbow();
    // Rainbow có cả từ phủ định/mức độ/cảm xúc, nhưng chúng quyết định sắc thái review ("not good", "the best") nên giữ lại
    private static final Set<String> KEEP_WORDS = Set.of(
            "not", "no", "nor", "never", "none", "nothing", "neither", "nobody", "cannot", "without", "against", "hardly",
            "but", "however", "very", "too", "only", "least", "less", "more", "most", "much", "few", "enough", "really",
            "best", "better", "well", "like", "liked", "thanks", "thank", "appreciate", "useful", "wish",
            "unfortunately", "hopefully", "definitely", "ok", "okay");
    private static final Pattern WORD = Pattern.compile("[A-Za-z]+(?:'[A-Za-z]+)*");   // giữ nguyên "don't", "isn't"

    /** Bỏ stopword (không phân biệt hoa thường), giữ nguyên dấu câu và các từ còn lại. */
    private static String removeStopwords(String s) {
        String straightQuotes = s.replace('’', '\'').replace('‘', '\'');   // "don’t" -> "don't" để không bị tách đôi
        String kept = WORD.matcher(straightQuotes).replaceAll(m -> {
            String w = m.group().toLowerCase(Locale.ROOT);
            return STOPWORDS.isStopword(w) && !KEEP_WORDS.contains(w) ? "" : Matcher.quoteReplacement(m.group());
        });
        return kept.replaceAll("\\s+", " ").strip();
    }

    public static void main(String[] args) throws Exception {
        convert(
                Path.of("raw-data", "Health_and_Personal_Care.jsonl"),
                Path.of("datasets", "Health_and_Personal_Care.arff"),
                25_000, // số dòng lấy mẫu
                42);    // seed cố định để lần nào chạy cũng ra cùng kết quả
    }
}
