# n8n Workflow Build Notes

## Credentials
1. Create Google Sheets OAuth2 credentials in n8n using your Google Client ID and Client Secret.
2. Add the redirect URI shown by n8n into Google Cloud OAuth credentials.
3. Authorize the Google account that owns the sheet.

## Recommended workflow
1. Trigger: Cron (every 1 minute)
2. Google Sheets: Read rows from sheet
3. Code node: filter rows where `status` is `pending`
4. Split In Batches: process one row at a time
5. HTTP Request: POST row to `http://video-processor:8000/process`
6. IF node: check `success === true`
7. Google Sheets Update Row (success branch)
8. Google Sheets Update Row (failure branch)

## Code node sample
```javascript
const rows = $input.all();
return rows.filter(item => {
  const status = String(item.json.status || '').trim().toLowerCase();
  return status === 'pending';
});
```

## HTTP request body
```json
{
  "job_id": "={{ $json.job_id }}",
  "mode": "={{ ($json.mode || 'auto').toLowerCase() }}",
  "youtube_url": "={{ $json.youtube_url }}",
  "start_time": "={{ $json.start_time || null }}",
  "end_time": "={{ $json.end_time || null }}",
  "desired_duration": "={{ Number($json.desired_duration || 30) }}",
  "language": "={{ $json.language || 'en' }}",
  "subtitle_enabled": "={{ $json.subtitle_enabled === true || String($json.subtitle_enabled).toUpperCase() === 'TRUE' }}",
  "cta_enabled": "={{ $json.cta_enabled === true || String($json.cta_enabled).toUpperCase() === 'TRUE' }}",
  "watermark_enabled": "={{ $json.watermark_enabled === true || String($json.watermark_enabled).toUpperCase() === 'TRUE' }}",
  "watermark_text": "={{ $json.watermark_text || '' }}"
}
```
