"""
MovieShort AI — Thread-local clip-label prefixing for console output.

process_multiple() renders several clips in parallel worker threads
(ThreadPoolExecutor). Without a per-line label, their interleaved print()
output is impossible to attribute to one clip or another — e.g. two clips'
"Person scan: N%" lines interleave with no way to tell which clip either one
belongs to.

log() reads a thread-local "current clip" label set once by process_clip()
at the start of a clip's work, and prefixes every line with it. Any function
called from within that clip's processing — however deep the call chain —
gets correctly labeled output for free, as long as it calls log() instead of
a raw print(). A plain print() (no label set, e.g. batch-level analysis
messages that run on the main thread before/after per-clip rendering) is
unaffected: log() with no active label behaves exactly like print().

The one place a label has to be propagated explicitly rather than inherited
automatically is across an actual new thread boundary — Python thread-locals
don't carry into a child thread. core/processor.py's person-scan timeout
wrapper spawns exactly one such child thread; it passes the label through as
a plain argument and re-sets it there (see analyze_persons()).
"""
import threading

_context = threading.local()


def set_clip_label(label):
    """Set the label log() prefixes every line with, in the current thread."""
    _context.label = label


def clear_clip_label():
    _context.label = None


def get_clip_label():
    """Current thread's label, or None if not inside labeled clip processing."""
    return getattr(_context, "label", None)


def log(message=""):
    """print(), prefixed with the current thread's clip label (if any)."""
    label = get_clip_label()
    if not label:
        print(message)
        return
    text = str(message)
    if not text:
        print(f"[{label}]")
        return
    prefixed = "\n".join(
        f"[{label}] {line}" if line else line
        for line in text.split("\n")
    )
    print(prefixed)
