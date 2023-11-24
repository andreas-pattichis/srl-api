SELECT a.id, a.user_id, a.save_time, a.username, a.url, a.essay_content, a.essay_content_json
FROM essay a
INNER JOIN (
    SELECT user_id, MAX(save_time) save_time
    FROM essay
    GROUP BY user_id
) b ON a.user_id = b.user_id AND a.save_time = b.save_time
LIMIT 1;