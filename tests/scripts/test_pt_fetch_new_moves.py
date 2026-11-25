from scripts.pt_fetch_new_moves import parse_move_page

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>Bridge Flip - Parkour Theory</title>
<meta name="description" content="Down Monkey set from bridge position. First landed in 2017 &amp; filmed.">
<meta property="og:title" content="Bridge Flip - Parkour Theory">
<meta property="og:image" content="https://customer-b2cflnmq20t5mr78.cloudflarestream.com/502032a4343508e62de10929f81ef682/thumbnails/thumbnail.jpg">
<meta property="og:url" content="https://www.parkourtheory.com/move?id=572ed8bc-6069-41dd-96f7-1efcfed36191">
</head><body><div id="root"></div></body></html>"""


def test_parse_move_page():
    assert parse_move_page(PAGE) == {
        "name": "Bridge Flip",
        "description": "Down Monkey set from bridge position. First landed in 2017 & filmed.",
        "video_url": "https://customer-b2cflnmq20t5mr78.cloudflarestream.com/502032a4343508e62de10929f81ef682/downloads/default.mp4",
        "pt_id": "572ed8bc-6069-41dd-96f7-1efcfed36191",
    }


def test_parse_move_page_without_video():
    move = parse_move_page('<meta property="og:title" content="Kong Vault - Parkour Theory">')
    assert move["name"] == "Kong Vault"
    assert move["video_url"] == "" and move["pt_id"] == "" and move["description"] == ""
