"""Synthetic memory workload (DATA-02 / DATA-04): seeded, deterministic, leak-free.

World (UnSeenTimeQA / LoCoMo patterns)
  persons, projects, orgs with invented names; attributes with timelines
  (supersede = the fact changed; correct = an earlier statement was wrong);
  person "twins" with near-identical names; an event graph of meetings that
  mentions several entities at once.
Notes
  one note per statement, grouped into daily sessions (session_id d0000..),
  role "user", entity_tags = entities mentioned. Explicit updates say "changed
  from A to B" and carry supersedes_note_id; implicit ones restate the new value
  without reference (supersedes_note_id stays None -- the system must infer it).
Distractor tiers (relative to a question about entity E, attribute A)
  filler (no entity) / near_dup (E's twin, same A) / same_entity (E, other A,
  incl. events) / superseded (E, A, older value) / other (unrelated entities).
Queries (VersionRAG taxonomy)
  exact_lookup, semantic_recall, latest_value, point_in_time, what_changed, multi_hop.
Streams
  notes.jsonl     = base corpus (ts < split_ts)
  updates.jsonl   = every later note, as add / supersede / correct ops in ts order
  notes_final.jsonl = everything with ground-truth valid_to for explicit chains (static runs)
  questions.jsonl = end-of-stream questions (static); questions_stream.jsonl = questions
                    spread over the stream with the query mix shifting at the midpoint
Ground truth not shown to systems lives in data/_audit/<name>/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from mem27.vschema import Note, Question, Update, write_jsonl

# ------------------------------------------------------------------ config

UPDATE_RATES = {"none": 0.0, "low": 0.25, "medium": 0.8, "high": 2.0}  # mean value changes per mutable timeline
QUERY_TYPES = ["exact_lookup", "semantic_recall", "latest_value", "point_in_time", "what_changed", "multi_hop"]


@dataclass
class SynthConfig:
    name: str = "synth_10k"
    n_notes: int = 10_000
    seed: int = 0
    update_rate: str = "medium"
    start: str = "2024-01-01"
    days: int = 365
    base_frac: float = 0.5
    filler_frac: float = 0.3
    twin_frac: float = 0.25
    explicit_frac: float = 0.5
    correct_frac: float = 0.2
    restate_rate: float = 0.4
    n_questions: int = 600
    dev_frac: float = 0.2
    mix: dict = field(default_factory=lambda: {"exact_lookup": 1, "semantic_recall": 1, "latest_value": 1,
                                               "point_in_time": 1, "what_changed": 1, "multi_hop": 1})
    stream_mix_before: dict = field(default_factory=lambda: {"latest_value": 0.7, "point_in_time": 0.3})
    stream_mix_after: dict = field(default_factory=lambda: {"latest_value": 0.3, "point_in_time": 0.7})
    n_stream_questions: int = 400

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynthConfig":
        import yaml
        return cls(**yaml.safe_load(Path(path).read_text()))


# ------------------------------------------------------------------ vocab

SYL = ("ka ve lo mir tha dor quin bel zar ris tov nel sha pra vik ul eth or yan fel gor lis mon tes rav ost cal "
       "pen dra sel bru cor dun ix jo kel lum nor pha rho sti tur vam wex xan yor zem ael bri cha dav esk fio "
       "gal hum ilo jas kru lev mak nym obr pel qua ros sav tyr umb vel wyn").split()
ROLES = ["data engineer", "product manager", "UX researcher", "site reliability engineer", "account executive",
         "technical writer", "security analyst", "QA lead", "solutions architect", "recruiter", "controller",
         "ML engineer", "support manager", "growth analyst", "hardware designer", "legal counsel",
         "operations lead", "frontend developer", "backend developer", "data scientist"]
STATUSES = ["in planning", "active", "paused", "in review", "shipped", "cancelled"]
ORG_SFX = ["Labs", "Collective", "Works", "Systems", "Holdings", "Foundry", "Analytics", "Dynamics"]
PLACE_SFX = ["ford", "mere", " Bay", " Heights", "wick", " Falls", "port", " Vale"]
FOODS = ["noodles", "a burrito", "lentil soup", "sushi", "a salad", "dumplings", "pizza", "a sandwich"]
ADJS = ["great", "too salty", "fine", "surprisingly good", "forgettable", "cold", "excellent"]
WEATHER = ["rain", "sunshine", "fog", "wind", "light snow", "heavy clouds"]
TOPICS = ["sourdough baking", "bird migration", "chess openings", "home espresso", "trail running",
          "watercolor painting", "jazz history", "houseplants", "budget travel", "board games"]
FILLER = [
    "Had {food} for lunch today; it was {adj}.",
    "Weather check: {weather} most of the afternoon.",
    "Spent the evening reading about {topic}.",
    "The office printer jammed again this morning.",
    "Note to self: renew the parking permit before the end of the month.",
    "Watched a documentary about {topic} tonight.",
    "Reminder to water the plants on the balcony.",
    "Long standup today, nothing major to report.",
    "Tried a new route to work; it took {n} minutes.",
    "Ordered {food} for the team meeting; everyone said it was {adj}.",
    "Need to pick up batteries and coffee filters this week.",
    "Booked a dentist appointment for {n} days from now.",
]

# attribute specs: value kind, mutability (relative change propensity), templates
ATTRS = {
    "person": {
        "employer": dict(kind="org", mut=1.0,
            say=["{E} works at {V}.", "{E} is employed by {V}.", "{E}'s employer is {V}."],
            exp=["{E} left {O} and joined {V}.", "{E}'s employer changed from {O} to {V}."],
            imp=["{E} started a new job at {V}.", "These days {E} works at {V}."],
            cor=["Correction: {E} works at {V}, not {O}."],
            q_latest=["Where does {E} work now?"], q_sem=["Which organization currently employs {E}?"],
            q_pit=["Where did {E} work on {D}?"], q_chg=["Where did {E} work before their current job, and where do they work now?"]),
        "city": dict(kind="place", mut=0.8,
            say=["{E} lives in {V}.", "{E} is based in {V}.", "{E}'s home is in {V}."],
            exp=["{E} moved from {O} to {V}.", "{E}'s city changed from {O} to {V}."],
            imp=["{E} just settled into a new apartment in {V}.", "{E} lives in {V} now."],
            cor=["Correction: {E} lives in {V}, not {O}."],
            q_latest=["Which city does {E} live in now?"], q_sem=["Where is {E}'s home these days?"],
            q_pit=["Which city did {E} live in on {D}?"], q_chg=["Where did {E} live before moving, and where do they live now?"]),
        "role": dict(kind="role", mut=0.6,
            say=["{E} is a {V}.", "{E} works as a {V}.", "{E}'s job title is {V}."],
            exp=["{E} switched from {O} to {V}.", "{E}'s role changed from {O} to {V}."],
            imp=["{E} was promoted and is now a {V}.", "{E} now works as a {V}."],
            cor=["Correction: {E} is a {V}, not a {O}."],
            q_latest=["What is {E}'s job title now?"], q_sem=["What kind of work does {E} do these days?"],
            q_pit=["What was {E}'s job title on {D}?"], q_chg=["What was {E}'s previous role and what is it now?"]),
        "project": dict(kind="project", mut=1.0,
            say=["{E} is assigned to {V}.", "{E} works on {V}.", "{E}'s current project is {V}."],
            exp=["{E} moved from {O} to {V}.", "{E}'s project changed from {O} to {V}."],
            imp=["{E} was staffed on {V}.", "{E} spends all week on {V} now."],
            cor=["Correction: {E} is on {V}, not {O}."],
            q_latest=["Which project is {E} assigned to now?"], q_sem=["What is {E} busy building at the moment?"],
            q_pit=["Which project was {E} assigned to on {D}?"], q_chg=["Which project was {E} on before, and which one now?"]),
        "manager": dict(kind="person", mut=0.7,
            say=["{E} reports to {V}.", "{E}'s manager is {V}.", "{V} manages {E}."],
            exp=["{E}'s manager changed from {O} to {V}.", "{E} now reports to {V} instead of {O}."],
            imp=["{E} had a first one-on-one with new manager {V}.", "{E} reports to {V} these days."],
            cor=["Correction: {E} reports to {V}, not {O}."],
            q_latest=["Who is {E}'s manager now?"], q_sem=["Who does {E} answer to at work?"],
            q_pit=["Who was {E}'s manager on {D}?"], q_chg=["Who was {E}'s previous manager and who is it now?"]),
        "extension": dict(kind="ext", mut=0.2,
            say=["{E}'s desk extension is {V}.", "You can reach {E} at extension {V}.", "{E}'s phone extension is {V}."],
            exp=["{E}'s extension changed from {O} to {V}."], imp=["New extension for {E}: {V}."],
            cor=["Correction: {E}'s extension is {V}, not {O}."],
            q_latest=["What is {E}'s desk extension?"], q_sem=["Which number do I dial to reach {E}'s desk?"],
            q_pit=["What was {E}'s extension on {D}?"], q_chg=["What was {E}'s old extension and what is the new one?"]),
        "tool": dict(kind="tool", mut=0.8,
            say=["{E}'s go-to tool is {V}.", "{E} does most work in {V}.", "{E} swears by {V}."],
            exp=["{E} switched from {O} to {V}.", "{E}'s main tool changed from {O} to {V}."],
            imp=["{E} has been raving about {V} lately.", "{E} does everything in {V} now."],
            cor=["Correction: {E}'s main tool is {V}, not {O}."],
            q_latest=["What is {E}'s main tool now?"], q_sem=["Which software does {E} rely on day to day?"],
            q_pit=["What was {E}'s main tool on {D}?"], q_chg=["Which tool did {E} use before, and which now?"]),
    },
    "project": {
        "lead": dict(kind="person", mut=0.8,
            say=["{E} is led by {V}.", "{V} leads {E}.", "The lead on {E} is {V}."],
            exp=["Leadership of {E} passed from {O} to {V}.", "{E}'s lead changed from {O} to {V}."],
            imp=["{V} took the helm of {E}.", "{V} runs {E} now."],
            cor=["Correction: {E} is led by {V}, not {O}."],
            q_latest=["Who leads {E} now?"], q_sem=["Who is in charge of {E}?"],
            q_pit=["Who led {E} on {D}?"], q_chg=["Who led {E} before, and who leads it now?"]),
        "status": dict(kind="status", mut=1.5,
            say=["{E} is {V}.", "Status of {E}: {V}.", "{E} is currently {V}."],
            exp=["{E} went from {O} to {V}.", "{E}'s status changed from {O} to {V}."],
            imp=["{E} is {V} as of this week.", "Latest on {E}: {V}."],
            cor=["Correction: {E} is {V}, not {O}."],
            q_latest=["What is the status of {E} now?"], q_sem=["How is {E} doing right now?"],
            q_pit=["What was the status of {E} on {D}?"], q_chg=["What was {E}'s previous status and what is it now?"]),
        "budget": dict(kind="budget", mut=0.2,
            say=["{E}'s budget code is {V}.", "Charge {E} work to {V}.", "{E} bills to {V}."],
            exp=["{E}'s budget code changed from {O} to {V}."], imp=["New budget code for {E}: {V}."],
            cor=["Correction: {E}'s budget code is {V}, not {O}."],
            q_latest=["What is {E}'s budget code?"], q_sem=["Which code should {E} expenses be billed to?"],
            q_pit=["What was {E}'s budget code on {D}?"], q_chg=["What was {E}'s old budget code and what is the new one?"]),
    },
    "org": {
        "hq": dict(kind="place", mut=0.3,
            say=["{E} is headquartered in {V}.", "{E}'s head office is in {V}."],
            exp=["{E} moved its headquarters from {O} to {V}."], imp=["{E} opened its new head office in {V}."],
            cor=["Correction: {E} is headquartered in {V}, not {O}."],
            q_latest=["Where is {E} headquartered now?"], q_sem=["In which city is {E}'s main office?"],
            q_pit=["Where was {E} headquartered on {D}?"], q_chg=["Where was {E}'s headquarters before, and where is it now?"]),
        "ceo": dict(kind="person", mut=0.4,
            say=["{E}'s CEO is {V}.", "{V} runs {E} as chief executive."],
            exp=["{E}'s CEO changed from {O} to {V}."], imp=["{V} was named chief executive of {E}."],
            cor=["Correction: {E}'s CEO is {V}, not {O}."],
            q_latest=["Who is the CEO of {E} now?"], q_sem=["Who is at the top of {E}?"],
            q_pit=["Who was the CEO of {E} on {D}?"], q_chg=["Who was {E}'s previous CEO and who is it now?"]),
    },
}
EXACT_ATTRS = {("person", "extension"), ("project", "budget")}
MULTI_HOP = [  # (type, attr1) -> value entity's (type2, attr2)
    (("person", "project"), ("project", "lead"), "Who leads the project that {E} is assigned to?"),
    (("person", "employer"), ("org", "hq"), "In which city is the headquarters of the company {E} works for?"),
    (("person", "manager"), ("person", "city"), "Which city does {E}'s manager live in?"),
]
PREFIXES = ["", "", "Reminder: ", "As mentioned before, ", "Just noting that ", "FYI, "]
EVENT = ["{A} and {B} met in {P} to discuss {J}.", "{A} presented {J} to {B} in {P}.",
         "{A} and {B} had coffee in {P} and talked about {J}."]


# ------------------------------------------------------------------ world

class Namer:
    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def word(self, lo=2, hi=3) -> str:
        return "".join(self.rng.choice(SYL) for _ in range(self.rng.randint(lo, hi))).capitalize()

    def unique(self, fn) -> str:
        for _ in range(1000):
            s = fn()
            if s.lower() not in self.used:
                self.used.add(s.lower())
                return s
        raise RuntimeError("name space exhausted")


@dataclass
class Fact:  # one assertion note's ground truth
    note_id: str
    entity: str
    etype: str
    attr: str
    value: str
    ts: str
    kind: str  # initial | restate | supersede | correct
    explicit: bool | None
    version: int
    prev_note_id: str | None  # ground-truth predecessor (also for implicit)


class Synth:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.namer = Namer(self.rng)
        self.t0 = datetime.fromisoformat(cfg.start)
        self.end = self.t0 + timedelta(days=cfg.days)
        self.split_ts = self.t0 + timedelta(days=cfg.days * cfg.base_frac)
        self.raw: list[dict] = []  # notes before session assignment: {ts, text, tags, fact|None, kind}
        self.timelines: dict[tuple[str, str], list[Fact]] = {}
        self.entities: dict[str, list[str]] = defaultdict(list)
        self.twin_of: dict[str, str] = {}
        self.fact_by_id: dict[str, Fact] = {}

    # -- sizing
    def sizes(self) -> dict:
        fact_notes = self.cfg.n_notes * (1 - self.cfg.filler_frac)
        mean_chg = UPDATE_RATES[self.cfg.update_rate]
        per_tl = (1 + 0.7 * mean_chg) * (1 + self.cfg.restate_rate)
        per_person = len(ATTRS["person"]) * per_tl + 0.3 * len(ATTRS["project"]) * per_tl \
            + 0.08 * len(ATTRS["org"]) * per_tl + 0.6
        n_person = max(12, round(fact_notes / per_person))
        return {"person": n_person, "project": max(4, round(0.3 * n_person)), "org": max(3, round(0.08 * n_person)),
                "place": max(8, round(0.12 * n_person)), "tool": max(8, round(0.06 * n_person))}

    def build_world(self) -> None:
        s = self.sizes()
        nm = self.namer
        self.pools = {
            "place": [nm.unique(lambda: nm.word(2, 2) + self.rng.choice(PLACE_SFX)) for _ in range(s["place"])],
            "tool": [nm.unique(lambda: f"{nm.word(2, 3)} {self.rng.randint(2, 9)}") for _ in range(s["tool"])],
            "role": ROLES, "status": STATUSES,
        }
        for _ in range(s["org"]):
            self.entities["org"].append(nm.unique(lambda: f"{nm.word(2, 3)} {self.rng.choice(ORG_SFX)}"))
        for _ in range(s["project"]):
            self.entities["project"].append(nm.unique(lambda: f"Project {nm.word(2, 3)}"))
        n_twins = round(s["person"] * self.cfg.twin_frac / (1 + self.cfg.twin_frac))
        for _ in range(s["person"] - n_twins):
            self.entities["person"].append(nm.unique(lambda: f"{nm.word(2, 3)} {nm.word(2, 3)}"))
        bases = self.rng.sample(self.entities["person"], min(n_twins, len(self.entities["person"])))
        for b in bases:  # same first name, different surname
            first = b.split()[0]
            t = nm.unique(lambda: f"{first} {nm.word(2, 3)}")
            self.entities["person"].append(t)
            self.twin_of[t], self.twin_of[b] = b, t
        self.pools.update({"org": self.entities["org"], "project": self.entities["project"],
                           "person": self.entities["person"]})

    # -- values
    def draw(self, kind: str, entity: str, avoid: str | None = None) -> str:
        for _ in range(50):
            if kind == "ext":
                v = str(self.rng.randint(1000, 9999))
            elif kind == "budget":
                v = f"BC-{self.rng.randint(1000, 9999)}"
            else:
                v = self.rng.choice(self.pools[kind])
            if v != avoid and v != entity:
                return v
        return v

    def t_rand(self, lo: datetime, hi: datetime) -> datetime:
        span = max(1.0, (hi - lo).total_seconds())
        t = lo + timedelta(seconds=self.rng.random() * span)
        return t.replace(hour=self.rng.randint(8, 20), minute=self.rng.randint(0, 59), second=0, microsecond=0) \
            if lo.date() != hi.date() else t.replace(microsecond=0)

    # -- timelines
    def make_timelines(self) -> None:
        mean_chg = UPDATE_RATES[self.cfg.update_rate]
        for etype in ("person", "project", "org"):
            for e in self.entities[etype]:
                for attr, spec in ATTRS[etype].items():
                    early = self.rng.random() < 0.75
                    hi = self.split_ts if early else self.end - timedelta(days=self.cfg.days * 0.05)
                    t = self.t_rand(self.t0, hi)
                    n_chg = self._poisson(mean_chg * spec["mut"]) if mean_chg else 0
                    times = sorted(self.t_rand(t + timedelta(days=1), self.end - timedelta(days=1)) for _ in range(n_chg))
                    times = [x for x in times if x > t]
                    tl: list[Fact] = []
                    val = self.draw(spec["kind"], e)
                    tl.append(self._fact(e, etype, attr, val, t, "initial", None, 1, None))
                    for tc in times:
                        is_cor = self.rng.random() < self.cfg.correct_frac
                        explicit = self.rng.random() < self.cfg.explicit_frac
                        new = self.draw(spec["kind"], e, avoid=tl[-1].value)
                        tl.append(self._fact(e, etype, attr, new, tc, "correct" if is_cor else "supersede",
                                             explicit, tl[-1].version + 1, tl[-1].note_id))
                    # restatements of each version, inside its validity window
                    for i, f in enumerate(list(tl)):
                        lo = datetime.fromisoformat(f.ts)
                        hi_v = datetime.fromisoformat(tl[i + 1].ts) if i + 1 < len(tl) else self.end
                        for _ in range(self._poisson(self.cfg.restate_rate)):
                            if (hi_v - lo).total_seconds() > 2 * 86400:
                                tr = self.t_rand(lo + timedelta(days=1), hi_v - timedelta(days=1))
                                self._fact(e, etype, attr, f.value, tr, "restate", None, f.version, f.note_id,
                                           store=tl, text_of=f)
                    tl.sort(key=lambda f: f.ts)
                    self.timelines[(e, attr)] = tl

    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1

    def _fact(self, e, etype, attr, val, t: datetime, kind, explicit, version, prev, store=None, text_of=None) -> Fact:
        spec = ATTRS[etype][attr]
        nid = f"n{len(self.raw):07d}"  # provisional, renamed after session assignment
        if kind in ("initial",):
            text = self.rng.choice(spec["say"])
        elif kind == "restate":
            text = self.rng.choice(PREFIXES) + self.rng.choice(spec["say"])
        elif explicit:
            text = self.rng.choice(spec["cor"] if kind == "correct" else spec["exp"])
        else:
            text = self.rng.choice(spec["imp"] if kind == "supersede" else spec["say"])
        old = None
        if "{O}" in text:  # previous version's value
            old = self.fact_by_id[prev].value
            text = text.replace("{O}", old)
        text = text.replace("{E}", e).replace("{V}", val)
        if text[0].islower():
            text = text[0].upper() + text[1:]
        f = Fact(nid, e, etype, attr, val, t.isoformat(), kind, explicit, version, prev)
        tags = [e, val] if spec["kind"] in ("org", "project", "person") else [e]
        if old and spec["kind"] in ("org", "project", "person"):
            tags.append(old)
        self.raw.append({"ts": t.isoformat(), "text": text, "tags": tags, "fact": f, "kind": kind})
        self.fact_by_id[nid] = f
        if store is not None:
            store.append(f)
        return f

    def make_events(self) -> None:
        persons, projects, places = self.entities["person"], self.entities["project"], self.pools["place"]
        for _ in range(round(0.6 * len(persons))):
            a, b = self.rng.sample(persons, 2)
            j, p = self.rng.choice(projects), self.rng.choice(places)
            t = self.t_rand(self.t0, self.end - timedelta(days=1))
            text = self.rng.choice(EVENT).format(A=a, B=b, P=p, J=j)
            self.raw.append({"ts": t.isoformat(), "text": text, "tags": [a, b, j], "fact": None, "kind": "event"})

    def make_filler(self) -> None:
        n_fill = max(0, self.cfg.n_notes - len(self.raw))
        for _ in range(n_fill):
            t = self.t_rand(self.t0, self.end - timedelta(days=1))
            text = self.rng.choice(FILLER).format(food=self.rng.choice(FOODS), adj=self.rng.choice(ADJS),
                                                  weather=self.rng.choice(WEATHER), topic=self.rng.choice(TOPICS),
                                                  n=self.rng.randint(5, 90))
            self.raw.append({"ts": t.isoformat(), "text": text, "tags": [], "fact": None, "kind": "filler"})

    # -- notes
    def assign_notes(self) -> list[Note]:
        order = sorted(range(len(self.raw)), key=lambda i: (self.raw[i]["ts"], i))
        rename, per_day = {}, Counter()
        notes = []
        hid = self.cfg.name
        for i in order:
            r = self.raw[i]
            day = (datetime.fromisoformat(r["ts"]).date() - self.t0.date()).days
            sid = f"d{day:04d}"
            turn = per_day[sid]
            per_day[sid] += 1
            nid = f"{hid}:{sid}:{turn}"
            old = r["fact"].note_id if r["fact"] else f"x{i}"
            rename[old] = nid
            r["note_id"] = nid
        for tl in self.timelines.values():
            for f in tl:
                f.note_id = rename[f.note_id]
                f.prev_note_id = rename.get(f.prev_note_id) if f.prev_note_id else None
        for i in order:
            r = self.raw[i]
            f = r["fact"]
            sup = f.prev_note_id if (f and f.kind in ("supersede", "correct") and f.explicit) else None
            notes.append(Note(note_id=r["note_id"], history_id=hid, session_id=r["note_id"].split(":")[1],
                              turn_idx=int(r["note_id"].rsplit(":", 1)[1]), role="user", text=r["text"],
                              ts=r["ts"], version=f.version if f else 1, supersedes_note_id=sup,
                              metadata={}, entity_tags=sorted(set(r["tags"]))))
        return notes

    # -- truth queries
    def value_at(self, e: str, attr: str, t: str) -> tuple[Fact | None, list[Fact]]:
        """Version in force at time t (corrections are retroactive), and the facts asserting it."""
        tl = [f for f in self.timelines[(e, attr)] if f.ts <= t]
        if not tl:
            return None, []
        versions = [f for f in tl if f.kind != "restate"]
        cur = versions[-1]
        support = [f for f in tl if f.version == cur.version]
        return cur, support

    def generate(self) -> dict:
        self.build_world()
        self.make_timelines()
        self.make_events()
        self.make_filler()
        notes = self.assign_notes()
        return {"notes": notes}


# ------------------------------------------------------------------ questions

def _qdate(t: datetime) -> str:
    return t.replace(microsecond=0).isoformat()


def make_questions(S: Synth, n: int, mix: dict, at: list[datetime], rng: random.Random, prefix: str,
                   split_seed: int) -> list[dict]:
    keys = sorted(S.timelines)
    types = [t for t in QUERY_TYPES if mix.get(t, 0) > 0]
    tot = sum(mix[t] for t in types)
    quota = {t: int(n * mix[t] / tot) for t in types}
    for t in sorted(types, key=lambda t: -(n * mix[t] / tot - quota[t]))[: n - sum(quota.values())]:
        quota[t] += 1
    out, seen = [], set()
    slots = [(t, i) for t in types for i in range(quota[t])]
    rng.shuffle(slots)  # interleave types so question times are not grouped by type
    for qt, _ in slots:
        for _try in range(400):
            qdate = at[len(out) % len(at)] if at else S.end
            qd = _qdate(qdate)
            q = _one_question(S, qt, keys, qd, rng)
            if q is None or (q["fact"]["entity"], q["fact"]["attr"], qt, qd[:10]) in seen:
                continue
            seen.add((q["fact"]["entity"], q["fact"]["attr"], qt, qd[:10]))
            q["question_id"] = f"{prefix}{len(out):05d}"
            q["question_date"] = qd
            out.append(q)
            break
    # stratified dev split
    by_t = defaultdict(list)
    for q in out:
        by_t[q["question_type"]].append(q)
    r2 = random.Random(split_seed)
    for t, qs in sorted(by_t.items()):
        r2.shuffle(qs)
        k = round(len(qs) * S.cfg.dev_frac)
        for i, q in enumerate(qs):
            q["split"] = "dev" if i < k else "test"
    return out


def _sess(nid: str) -> str:
    return nid.split(":")[1]


def _one_question(S: Synth, qt: str, keys, qd: str, rng: random.Random) -> dict | None:
    if qt == "multi_hop":
        (t1, a1), (t2, a2), tmpl = rng.choice(MULTI_HOP)
        e = rng.choice(S.entities[t1])
        if (e, a1) not in S.timelines:
            return None
        f1, sup1 = S.value_at(e, a1, qd)
        if f1 is None or (f1.value, a2) not in S.timelines:
            return None
        f2, sup2 = S.value_at(f1.value, a2, qd)
        if f2 is None:
            return None
        gold = [f.note_id for f in sup1 + sup2]
        return _q(qt, tmpl.replace("{E}", e), f2.value, gold, qd, e, a1,
                  {"hop1": f1.value, "hop2_entity": f1.value, "hop2_attr": a2}, S, qd, [])
    e, attr = rng.choice(keys)
    etype = next(t for t in ATTRS if attr in ATTRS[t] and e in S.entities[t])
    spec = ATTRS[etype][attr]
    tl = [f for f in S.timelines[(e, attr)] if f.ts <= qd]
    versions = [f for f in tl if f.kind != "restate"]
    if not versions:
        return None
    cur, support = S.value_at(e, attr, qd)
    stale = [f for f in tl if f.version != cur.version]
    stale_vals = sorted({f.value for f in stale if f.value != cur.value})
    if qt == "exact_lookup":
        if (etype, attr) not in EXACT_ATTRS:
            return None
        return _q(qt, rng.choice(spec["q_latest"]).replace("{E}", e), cur.value, [f.note_id for f in support],
                  qd, e, attr, {}, S, qd, stale_vals, cur)
    if qt == "semantic_recall":
        if (etype, attr) in EXACT_ATTRS:
            return None
        return _q(qt, rng.choice(spec["q_sem"]).replace("{E}", e), cur.value, [f.note_id for f in support],
                  qd, e, attr, {}, S, qd, stale_vals, cur)
    if qt == "latest_value":
        if len(versions) < 2:
            return None
        return _q(qt, rng.choice(spec["q_latest"]).replace("{E}", e), cur.value, [f.note_id for f in support],
                  qd, e, attr, {}, S, qd, stale_vals, cur)
    if qt == "point_in_time":
        sup_chain = [f for f in versions if f.kind != "correct"]
        if len(versions) < 2 or any(f.kind == "correct" for f in versions):
            return None  # corrections make "value on date D" ambiguous; keep pure supersede chains
        i = rng.randrange(len(sup_chain) - 1)
        lo = datetime.fromisoformat(sup_chain[i].ts) + timedelta(days=1)
        hi = datetime.fromisoformat(sup_chain[i + 1].ts) - timedelta(days=1)
        if hi <= lo:
            return None
        d = S.t_rand(lo, hi)
        at = _qdate(d)
        f, sup = S.value_at(e, attr, at)
        dtxt = d.strftime("%B %d, %Y").replace(" 0", " ")
        others = sorted({x.value for x in versions if x.value != f.value})
        return _q(qt, rng.choice(spec["q_pit"]).replace("{E}", e).replace("{D}", dtxt), f.value,
                  [x.note_id for x in sup], qd, e, attr, {"as_of_time": at}, S, at, others, f)
    if qt == "what_changed":
        if len(versions) < 2 or versions[-1].kind != "supersede":
            return None
        prev = versions[-2]
        prev_sup = [f for f in tl if f.version == prev.version]
        ans = f"from {prev.value} to {cur.value}"
        return _q(qt, rng.choice(spec["q_chg"]).replace("{E}", e), ans,
                  [f.note_id for f in support + prev_sup], qd, e, attr, {"old_value": prev.value, "new_value": cur.value},
                  S, qd, [], cur)
    return None


def _q(qt, text, answer, gold, qd, e, attr, extra, S: Synth, as_of, stale_vals, cur: Fact | None = None) -> dict:
    tl = S.timelines.get((e, attr), [])
    last_change = cur if cur is not None and cur.kind in ("supersede", "correct") else None
    fact = {"entity": e, "attr": attr, "value": answer if qt != "what_changed" else extra.get("new_value"),
            "stale_values": stale_vals, "n_versions": len({f.version for f in tl if f.ts <= as_of}),
            "latest_change_kind": last_change.kind if last_change else None,
            "latest_change_explicit": last_change.explicit if last_change else None,
            "superseded_note_ids": [f.note_id for f in tl if f.ts <= as_of and cur is not None
                                    and f.version != cur.version and f.note_id not in gold],
            "twin": S.twin_of.get(e)}
    fact.update(extra)
    return {"history_id": S.cfg.name, "question_type": qt, "question": text, "answer": answer,
            "gold_session_ids": sorted({_sess(g) for g in gold}), "gold_note_ids": sorted(set(gold)),
            "as_of_time": extra.get("as_of_time", qd), "fact": fact,
            "chance": _chance(S, e, attr, qt)}


def _chance(S: Synth, e: str, attr: str, qt: str) -> float:
    """P(correct) for a reader guessing uniformly from the right value pool with no memory."""
    if qt == "multi_hop" or qt == "what_changed":
        return 0.0
    etype = next(t for t in ATTRS if attr in ATTRS[t] and e in S.entities[t])
    kind = ATTRS[etype][attr]["kind"]
    if kind in ("ext", "budget"):
        return 1 / 9000
    return 1 / max(1, len(S.pools[kind]))


# ------------------------------------------------------------------ updates + writing

def note_tier_audit(S: Synth, notes: list[Note]) -> list[dict]:
    facts = {f.note_id: f for tl in S.timelines.values() for f in tl}
    rows = []
    for n in notes:
        f = facts.get(n.note_id)
        if f:
            rows.append({"note_id": n.note_id, "kind": f.kind, "entity": f.entity, "attr": f.attr, "value": f.value,
                         "version": f.version, "explicit": f.explicit, "prev_note_id": f.prev_note_id})
        else:
            rows.append({"note_id": n.note_id, "kind": "event" if n.entity_tags else "filler",
                         "entities": n.entity_tags})
    return rows


def build_updates(S: Synth, notes: list[Note]) -> tuple[list[Note], list[Update]]:
    facts = {f.note_id: f for tl in S.timelines.values() for f in tl}
    split = S.split_ts.isoformat()
    base = [n for n in notes if n.ts < split]
    ups = []
    for n in notes:
        if n.ts < split:
            continue
        f = facts.get(n.note_id)
        if f and f.kind in ("supersede", "correct"):
            op = f.kind
            explicit = bool(f.explicit)
            sup = f.prev_note_id if explicit else None
        else:
            op, explicit, sup = "add", False, None
        ups.append(Update(update_id=f"u{len(ups):07d}", ts=n.ts, op=op, note=asdict(n), supersedes_note_id=sup,
                          explicit=explicit))
    return base, ups


def final_snapshot(S: Synth, notes: list[Note]) -> list[Note]:
    """All notes; explicit supersessions close the predecessor's valid_to (what a system knows)."""
    by_id = {n.note_id: Note(**asdict(n)) for n in notes}
    for n in notes:
        if n.supersedes_note_id and n.supersedes_note_id in by_id:
            by_id[n.supersedes_note_id].valid_to = n.ts
    return [by_id[n.note_id] for n in notes]


def generate(cfg: SynthConfig, out_root: str | Path = "data") -> dict:
    S = Synth(cfg)
    notes = S.generate()["notes"]
    out = Path(out_root) / cfg.name
    audit = Path(out_root) / "_audit" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    base, ups = build_updates(S, notes)
    write_jsonl(out / "notes.jsonl", base)
    write_jsonl(out / "updates.jsonl", ups)
    write_jsonl(out / "notes_final.jsonl", final_snapshot(S, notes))
    rng = random.Random(cfg.seed + 1)
    static_q = make_questions(S, cfg.n_questions, cfg.mix, [S.end], rng, f"{cfg.name}_q", cfg.seed)
    write_jsonl(out / "questions.jsonl", static_q)
    # stream questions: spread over the update stream, mix shifts at the midpoint
    span = (S.end - S.split_ts).total_seconds()
    times = sorted(S.split_ts + timedelta(seconds=rng.random() * span) for _ in range(cfg.n_stream_questions))
    mid = S.split_ts + timedelta(seconds=span / 2)
    before = [t for t in times if t < mid]
    after = [t for t in times if t >= mid]
    sq = make_questions(S, len(before), cfg.stream_mix_before, before, rng, f"{cfg.name}_sb", cfg.seed) + \
        make_questions(S, len(after), cfg.stream_mix_after, after, rng, f"{cfg.name}_sa", cfg.seed)
    write_jsonl(out / "questions_stream.jsonl", sorted(sq, key=lambda q: q["question_date"]))
    write_jsonl(audit / "note_facts.jsonl", note_tier_audit(S, notes))
    with open(audit / "world.json", "w") as f:
        json.dump({"entities": S.entities, "twins": S.twin_of, "pools": {k: v for k, v in S.pools.items()
                                                                          if k in ("place", "tool")}}, f)
    stats = {
        "config": asdict(cfg), "n_notes_total": len(notes), "n_base_notes": len(base), "n_updates": len(ups),
        "ops": dict(Counter(u.op for u in ups)),
        "explicit_ops": dict(Counter(f"{u.op}:{u.explicit}" for u in ups if u.op != "add")),
        "note_kinds": dict(Counter(r["kind"] for r in S.raw)),
        "entities": {k: len(v) for k, v in S.entities.items()}, "twins": len(S.twin_of) // 2,
        "questions": dict(Counter(q["question_type"] for q in static_q)),
        "questions_split": dict(Counter(q["split"] for q in static_q)),
        "stream_questions": dict(Counter(q["question_type"] for q in sq)),
        "chance_accuracy_static": sum(q["chance"] for q in static_q) / max(1, len(static_q)),
        "split_ts": S.split_ts.isoformat(), "end": S.end.isoformat(),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-notes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--update-rate", default=None, choices=list(UPDATE_RATES))
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)
    cfg = SynthConfig.from_yaml(a.config) if a.config else SynthConfig()
    for k in ("name", "n_notes", "seed", "update_rate"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    print(json.dumps(generate(cfg, a.out), indent=2))


if __name__ == "__main__":
    main()
