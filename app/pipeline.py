from app.services import (
    classifier,
    extractor,
    reply_drafter,
    sentiment,
    summarizer,
    tagger,
    translator,
)


def process_ticket(ticket: dict) -> dict:
    language = translator.detect_language(ticket["body"])
    body_en = translator.translate(ticket["body"], "English")
    ticket = {**ticket, "body": body_en}

    category = classifier.classify_ticket(ticket["body"])
    mood = sentiment.get_sentiment(ticket["body"])
    tags = tagger.tag_ticket(ticket["body"])

    order_id = extractor.extract_order_id(ticket["body"])
    email = extractor.extract_email(ticket["body"])
    fields = extractor.extract_fields(ticket["body"])

    thread = ticket.get("thread", []) + [{"role": "customer", "content": ticket["body"]}]
    summary = summarizer.summarize_with_check(thread)
    thread_sentiments = sentiment.sentiment_per_message([m["content"] for m in thread])

    reply = reply_drafter.draft_reply(ticket)
    reply = translator.translate(reply, language)

    return {
        "id": ticket["id"],
        "language": language,
        "category": category,
        "sentiment": mood,
        "thread_sentiments": thread_sentiments,
        "tags": tags,
        "order_id": order_id,
        "email": email,
        "fields": fields,
        "summary": summary,
        "reply": reply,
    }
