"""
Thin Blogger API v3 wrapper, matching
google_client.py's shape (import the service builder, call a small
function, don't touch raw HTTP anywhere else).

Usage:
    from blogger_client import get_blog_id, create_post, list_posts, publish_post

    blog_id = get_blog_id()
    post = create_post(blog_id, "Title", "<p>Body as HTML.</p>", is_draft=True)
    # ... review post["url"] ...
    publish_post(blog_id, post["id"])
"""

from google_client import get_blogger_service

def _configured_url() -> str:
    from config_loader import load_config
    return str((load_config().get("google") or {}).get("blog_url") or "")


def get_blog_id(url: str | None = None) -> str:
    """Look up the blog's numeric ID by its URL (live API call, nothing hardcoded)."""
    url = url or _configured_url()
    if not url:
        raise RuntimeError('No blog set. Add {"google": {"blog_url": "https://yourblog.blogspot.com/"}} '
                           "to config.json, or pass the URL.")
    service = get_blogger_service()
    blog = service.blogs().getByUrl(url=url).execute()
    return blog["id"]


def list_posts(blog_id: str | None = None, max_results: int = 10) -> list[dict]:
    service = get_blogger_service()
    bid = blog_id or get_blog_id()
    resp = service.posts().list(blogId=bid, maxResults=max_results,
                                 fetchBodies=False).execute()
    return resp.get("items", [])


def create_post(blog_id: str, title: str, content_html: str,
                 is_draft: bool = True) -> dict:
    """Create a post. Draft by default -- a bad post going straight live
    with no review step is a worse failure mode than one extra API call to
    publish it once it's been read."""
    service = get_blogger_service()
    body = {"title": title, "content": content_html}
    return service.posts().insert(blogId=blog_id, body=body,
                                   isDraft=is_draft).execute()


def publish_post(blog_id: str, post_id: str) -> dict:
    """Flip an existing draft post live."""
    service = get_blogger_service()
    return service.posts().publish(blogId=blog_id, postId=post_id).execute()
