"""InferArena suite v2: the public prompt set (A = short chat, V = held-out tasks, R = headline copy-heavy tasks).

Written for the arena (public, nothing proprietary). Changing any prompt is a new suite version, never an edit to v2.
Prompts carry no per-run salt: identical prompts are what make the exact-output check possible. Prefix caching can only
help TTFT, so TTFT is taken from the first pass alone; decode speed is unaffected.
"""

_INVENTORY_PY = '''\
import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    sku: str
    name: str
    qty: int
    price: float
    tags: list = field(default_factory=list)


class Inventory:
    def __init__(self, path):
        self.path = path
        self.items = {}
        self.log = []

    def load(self):
        with open(self.path) as f:
            for row in json.load(f):
                self.items[row["sku"]] = Item(**row)

    def save(self):
        with open(self.path, "w") as f:
            json.dump([vars(i) for i in self.items.values()], f, indent=2)

    def add(self, sku, name, qty, price, tags=None):
        if sku in self.items:
            self.items[sku].qty += qty
        else:
            self.items[sku] = Item(sku, name, qty, price, tags or [])
        self.log.append((datetime.now(), "add", sku, qty))

    def remove(self, sku, qty):
        item = self.items[sku]
        if qty > item.qty:
            raise ValueError("not enough stock")
        item.qty -= qty
        if item.qty == 0:
            del self.items[sku]
        self.log.append((datetime.now(), "remove", sku, qty))

    def value(self):
        total = 0
        for i in self.items.values():
            total =+ i.qty * i.price
        return total

    def low_stock(self, threshold=5):
        return [i for i in self.items.values() if i.qty < threshold]

    def find(self, text):
        text = text.lower()
        return [i for i in self.items.values() if text in i.name.lower() or text in i.tags]

    def report(self):
        lines = []
        for i in sorted(self.items.values(), key=lambda x: x.name):
            lines.append(f"{i.sku:8} {i.name:30} {i.qty:5} {i.price:8.2f}")
        return "\\n".join(lines)
'''

_POLICY_MD = '''\
# Remote Work Policy

## 1. Scope
This policy applies to all full-time and part-time employees after their probation period (90 days).
Contractors follow the terms of their contract instead.

## 2. Eligibility
Employees may work remotely up to 3 days per week. Teams with on-site equipment (lab, warehouse, front desk)
may set a lower limit, which must be written in the team handbook. Requests for more than 3 days need approval
from both the line manager and HR, and are reviewed every 6 months.

## 3. Working hours
Core hours are 10:00 to 16:00 in the employee's local time. Outside core hours, people may arrange their own
schedule, but meetings scheduled at least 24 hours ahead must be attended or declined with a reason.

## 4. Equipment and expenses
The company provides a laptop, a monitor and a headset. Employees can claim up to 300 EUR once every two years
for a desk or chair, with a receipt. Internet costs are not reimbursed, except for on-call staff, who receive
30 EUR per month.

## 5. Security
Company data may only be accessed through the VPN. Printing confidential documents at home is not allowed.
Lost or stolen equipment must be reported to IT within 24 hours.

## 6. Travel to the office
Travel between home and the office is not paid. When an employee is asked to come in on a remote day for
less than 4 hours of work, the company pays public transport costs.
'''

_STATS_C = '''\
#include <stdlib.h>
#include <math.h>

typedef struct { double *v; size_t n; size_t cap; } Series;

int series_push(Series *s, double x) {
    if (s->n == s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 16;
        double *p = realloc(s->v, cap * sizeof(double));
        if (!p) return -1;
        s->v = p;
        s->cap = cap;
    }
    s->v[s->n++] = x;
    return 0;
}

double series_mean(const Series *s) {
    double sum = 0;
    for (size_t i = 0; i < s->n; i++) sum += s->v[i];
    return s->n ? sum / s->n : NAN;
}

double series_stddev(const Series *s) {
    if (s->n < 2) return NAN;
    double m = series_mean(s), acc = 0;
    for (size_t i = 0; i < s->n; i++) acc += (s->v[i] - m) * (s->v[i] - m);
    return sqrt(acc / (s->n - 1));
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double series_median(Series *s) {
    if (!s->n) return NAN;
    qsort(s->v, s->n, sizeof(double), cmp);
    return s->n % 2 ? s->v[s->n / 2] : (s->v[s->n / 2 - 1] + s->v[s->n / 2]) / 2;
}
'''

_SALES_CSV = '''\
region,quarter,product,units,revenue
North,Q1,Basic,1200,23880
North,Q1,Pro,340,33660
North,Q2,Basic,1310,26069
North,Q2,Pro,410,40590
South,Q1,Basic,980,19502
South,Q1,Pro,220,21780
South,Q2,Basic,1105,21989
South,Q2,Pro,265,26235
East,Q1,Basic,760,15124
East,Q1,Pro,510,50490
East,Q2,Basic,690,13731
East,Q2,Pro,560,55440
West,Q1,Basic,1430,28457
West,Q1,Pro,300,29700
West,Q2,Basic,1515,30149
West,Q2,Pro,330,32670
'''

_TOKENIZER_JS = '''\
export function tokenize(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\\s/.test(c)) { i++; continue; }
    if (/[0-9]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      out.push({ type: "num", value: parseFloat(src.slice(i, j)) });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[a-zA-Z0-9_]/.test(src[j])) j++;
      out.push({ type: "ident", value: src.slice(i, j) });
      i = j;
      continue;
    }
    if ("+-*/()=".includes(c)) { out.push({ type: "op", value: c }); i++; continue; }
    throw new Error(`unexpected character ${c} at ${i}`);
  }
  return out;
}
'''

_SERVER_LOG = '''\
2026-03-04 09:00:01 INFO  api      request GET /health 200 2ms
2026-03-04 09:00:03 INFO  api      request POST /orders 201 41ms
2026-03-04 09:00:04 WARN  db       slow query 812ms: SELECT * FROM orders WHERE customer_id = ?
2026-03-04 09:00:07 INFO  worker   job 5521 started (invoice-pdf)
2026-03-04 09:00:09 ERROR worker   job 5521 failed: font file not found: /srv/fonts/Inter.ttf
2026-03-04 09:00:09 INFO  worker   job 5521 retry 1/3 scheduled in 30s
2026-03-04 09:00:12 INFO  api      request GET /orders/88 200 7ms
2026-03-04 09:00:15 WARN  cache    hit rate below 60% (57%) for key prefix order:
2026-03-04 09:00:18 INFO  api      request POST /login 200 120ms
2026-03-04 09:00:21 ERROR api      request POST /payments 502 3004ms: upstream timeout (payments-gw)
2026-03-04 09:00:22 WARN  api      circuit breaker for payments-gw half-open
2026-03-04 09:00:39 INFO  worker   job 5521 retry 1 started
2026-03-04 09:00:40 ERROR worker   job 5521 failed: font file not found: /srv/fonts/Inter.ttf
2026-03-04 09:00:44 INFO  db       autovacuum on table orders finished in 3.1s
2026-03-04 09:00:51 ERROR api      request POST /payments 502 3001ms: upstream timeout (payments-gw)
2026-03-04 09:00:52 WARN  api      circuit breaker for payments-gw open
2026-03-04 09:01:02 INFO  api      request GET /health 200 2ms
2026-03-04 09:01:10 WARN  disk     /var/lib/app at 91% used
2026-03-04 09:01:15 INFO  worker   job 5522 started (email-receipt)
2026-03-04 09:01:16 INFO  worker   job 5522 done in 0.8s
2026-03-04 09:01:20 ERROR db       connection reset by peer (pool 3/20 busy)
2026-03-04 09:01:21 INFO  db       reconnected in 140ms
'''

SETS = {
    "A": {
        "max_tokens": 300, "context": 4096,
        "prompts": [
            ("a_heap", "Write a Python class that implements a min-heap with push, pop and peek methods. Include docstrings."),
            ("a_http2", "Explain how HTTP/2 multiplexing differs from HTTP/1.1 pipelining, and why it reduces latency."),
            ("a_mergesort", "Explain merge sort step by step, then give a short Python implementation."),
            ("a_books_json", "Generate a JSON array of 10 fictional books. Each item has title, author, year, genre and rating (1-5)."),
        ],
    },
    "V": {
        "max_tokens": 512, "context": 16384,
        "prompts": [
            ("v_bug_find", "Find the bugs in this Python module. For each bug, quote the line and explain the fix.\n\n" + _INVENTORY_PY),
            ("v_docstrings", "Add a concise docstring to every method of this class. Return the whole file.\n\n" + _INVENTORY_PY),
            ("v_rename", "Rename the class Inventory to StockBook and the attribute items to stock everywhere. Return the whole file.\n\n" + _INVENTORY_PY),
            ("v_tests", "Write pytest unit tests for add, remove and low_stock in this module.\n\n" + _INVENTORY_PY),
            ("v_list_fns", "List every function in this C file with its signature and a one-line description.\n\n" + _STATS_C),
            ("v_port_py", "Port this C code to idiomatic Python (a Series class with the same operations).\n\n" + _STATS_C),
            ("v_explain_c", "Explain what series_median does and point out any side effect a caller should know about.\n\n" + _STATS_C),
            ("v_policy_qa", "Using only this policy, answer: may an employee on probation work from home, and what can a remote employee claim for a chair?\n\n" + _POLICY_MD),
            ("v_policy_sum", "Summarise this policy in 6 bullet points for a new employee.\n\n" + _POLICY_MD),
            ("v_policy_table", "Turn the rules in this policy into a Markdown table with columns: topic, rule, limit or amount.\n\n" + _POLICY_MD),
            ("v_csv_totals", "From this CSV, compute total revenue per region and per product, and show both as Markdown tables.\n\n" + _SALES_CSV),
            ("v_csv_copy", "Copy this CSV exactly, but only the rows for the Pro product, and add a column price = revenue / units.\n\n" + _SALES_CSV),
            ("v_js_edit", "Add support for string literals in double quotes to this tokenizer (token type \"str\"). Return the whole file.\n\n" + _TOKENIZER_JS),
            ("v_js_explain", "Explain this tokenizer line by line for a junior developer.\n\n" + _TOKENIZER_JS),
        ],
    },
    # Headline copy-heavy tasks: most of the answer repeats the input, which is where speculative and copy-drafting
    # engines are fastest. Scored like A and V (InferArena rules, 2026-10-25).
    "R": {
        "max_tokens": 512, "context": 16384,
        "prompts": [
            ("r_rewrite_py", "Add type hints to every function and method in this module. Return the complete file.\n\n" + _INVENTORY_PY),
            ("r_edit_c", "Change series_push so the capacity grows by 1.5x instead of 2x, and return -1 when s is NULL. Return the complete file.\n\n" + _STATS_C),
            ("r_rename_c", "Rename the type Series to Samples and every series_ function to samples_. Return the complete file.\n\n" + _STATS_C),
            ("r_edit_js", "Make the tokenizer skip // line comments. Return the complete file.\n\n" + _TOKENIZER_JS),
            ("r_edit_doc", "Change the remote-work limit from 3 to 2 days per week and the chair allowance from 300 EUR to 400 EUR wherever they appear. Return the complete document.\n\n" + _POLICY_MD),
            ("r_edit_csv", "Add a column avg_price = revenue / units, rounded to 2 decimals, to every row. Return the complete CSV.\n\n" + _SALES_CSV),
            ("r_log_triage", "List every ERROR and WARN line of this log, grouped by component, with its timestamp.\n\n" + _SERVER_LOG),
        ],
    },
}
