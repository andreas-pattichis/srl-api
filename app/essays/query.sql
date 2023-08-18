SELECT a.id, a.user_id, a.save_time, a.username, a.url, a.essay_content, a.essay_content_json
FROM essay a
INNER JOIN (
    SELECT user_id, MAX(save_time) save_time
    FROM essay
    GROUP BY user_id
) b ON a.user_id = b.user_id AND a.save_time = b.save_time
LIMIT 1;
-- INTO OUTFILE '/data/home/sebram/out.txt';

INSERT INTO `essay` (`id`, `user_id`, `save_time`, `username`, `url`, `essay_content`, `essay_content_json`) VALUES
(1, '3', '1687866140519', 'fshl001 FSHL', 'https://nijmegen.floraproject.org/moodle/mod/page/view.php?id=7', 'I can write my essay here\r\nhet geeft alleen het aantal woorden aan. Is dat okï¿© of willen we weer aftellend?\r\nGeen NL versie??', '{\"ops\":[{\"insert\":\"I can write my essay here\\nhet geeft alleen het aantal woorden aan. Is dat okï¿© of willen we weer aftellend? \\nGeen NL versie??\\n\"}]}');
