"""
Test harness: stub streamlit + plotly + supabase + httpx completely,
pre-populate an admin session with fake MCR results, and exec() the
real app.py end-to-end to catch import/name/logic errors that ast.parse
cannot catch (undefined names, wrong function signatures, wrong dict
keys, etc.) — as close to a real run as possible without a live
Supabase/Streamlit server.

Buttons are forced to return True so every button-triggered code path
(including the new tabs) actually executes at least once.
"""
import sys, types, traceback, io, json
import numpy as np

# ═══════════════ Stub: streamlit ═══════════════
class _Ctx:
    """Stands in for st.columns()/st.expander()/st.container()/etc.
    Real Streamlit column/container objects support the SAME widget
    API as the top-level `st` module (e.g. `col1.number_input(...)`),
    so forward any missing attribute to the shared st stub instance."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        return getattr(st_stub, name)

class _FormCtx:
    """Stands in for the object returned by st.form(name) — tracks which
    form is currently open so form_submit_button() can tell which form
    it belongs to (real Streamlit ties submit buttons to their enclosing
    form; our stub needs the same to selectively trigger only 'add_ref'
    and not 'add_user'/'change_pw')."""
    def __init__(self, host, name):
        self.host = host
        self.name = name
    def __enter__(self):
        self.host._form_stack.append(self.name)
        return self
    def __exit__(self, *a):
        self.host._form_stack.pop()
        return False
    def __getattr__(self, name):
        return getattr(st_stub, name)

class _FakeUploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile: behaves like a file (BytesIO)
    but also carries a `.name` attribute that app.py checks (fn =
    ref_file.name.lower())."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name

class _SessionState(dict):
    pass

class _StStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.secrets = {"supabase": {"url": "https://fake.supabase.co", "key": "fake-key"}}
        self._form_stack = []

    # layout / containers
    def tabs(self, labels): return [_Ctx() for _ in labels]
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]
    def expander(self, *a, **kw): return _Ctx()
    def spinner(self, *a, **kw): return _Ctx()
    def container(self, *a, **kw): return _Ctx()

    # display (no-ops)
    def markdown(self, *a, **kw): pass
    def write(self, *a, **kw): pass
    def caption(self, *a, **kw): pass
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): print("  [st.error]", a[0] if a else "")
    def success(self, *a, **kw): pass
    def dataframe(self, *a, **kw): pass
    def plotly_chart(self, *a, **kw): pass
    def set_page_config(self, *a, **kw): pass
    def json(self, *a, **kw): pass
    def title(self, *a, **kw): pass
    def header(self, *a, **kw): pass
    def subheader(self, *a, **kw): pass
    def image(self, *a, **kw): pass
    def code(self, *a, **kw): pass

    # inputs — return sensible defaults so downstream code has valid types
    def _reg(self, kw, ret):
        """Real Streamlit auto-writes a keyed widget's return value into
        st.session_state[key]. Our stub must mimic this — several real
        app.py code paths (e.g. reading back st.session_state[f"identity_
        ref_select_{i}"] after the widget loop) rely on it, and silently
        returning a value without registering it caused those branches
        to look empty/skipped even though the widget "worked"."""
        key = kw.get("key")
        if key is not None:
            self.session_state[key] = ret
        return ret

    def text_input(self, label="", value="", **kw):
        # 'add_ref' form needs a non-empty compound name to pass its
        # own validation (`if not ref_name: st.error(...)`) — target
        # only that field by label text, everything else keeps the
        # normal (empty-by-default) behavior.
        if self._form_stack and self._form_stack[-1] == "add_ref" and \
           label in ("Nama senyawa", "Compound name"):
            return self._reg(kw, "Test Compound (harness)")
        return self._reg(kw, value)
    def text_area(self, label="", value="", **kw): return self._reg(kw, value)
    def number_input(self, label="", min_value=None, max_value=None, value=None, *a, **kw):
        if value is None:
            value = min_value if min_value is not None else 0
        return self._reg(kw, value)
    def checkbox(self, label="", value=False, **kw): return self._reg(kw, value)
    def selectbox(self, label="", options=(), index=0, format_func=None, **kw):
        options = list(options)
        if not options:
            return self._reg(kw, None)
        # Expander identity ("🎯 Chemical Identity Decision") menaruh
        # opsi placeholder (None, "— pilih reference —") di index 0 —
        # default_idx normal juga 0, jadi ref_id selalu None dan loopnya
        # `continue` (skip) walau library sudah diisi. Untuk key
        # identity_ref_select_* kita paksa pilih entri library nyata
        # (index 1 = seeded id=1) supaya jalur pearson/cosine benar2
        # tereksekusi dengan nilai nyata, bukan cuma cabang skip.
        key = kw.get("key", "") or ""
        if key.startswith("identity_ref_select_") and len(options) > 1:
            return self._reg(kw, options[1])
        idx = index if isinstance(index, int) and 0 <= index < len(options) else 0
        return self._reg(kw, options[idx])
    def multiselect(self, label="", options=(), default=None, **kw):
        return self._reg(kw, list(default) if default is not None else [])
    def slider(self, label="", min_value=0, max_value=1, value=None, **kw):
        return self._reg(kw, value if value is not None else min_value)
    def radio(self, label="", options=(), index=0, **kw):
        options = list(options)
        return self._reg(kw, options[index] if options else None)
    def file_uploader(self, label="", *a, **kw):
        # Only fake the ONE uploader we're deliberately exercising (the
        # library reference-spectrum upload inside the 'add_ref' form),
        # identified by its distinctive label text — every other
        # uploader (mixture data, JSON import, derivative ref, etc.)
        # keeps returning None so those untested flows stay untouched.
        if self._form_stack and self._form_stack[-1] == "add_ref":
            csv_bytes = _FAKE_REF_SPECTRUM_CSV.encode("utf-8")
            return _FakeUploadedFile(csv_bytes, "harness_test_ref.csv")
        return None
    def download_button(self, *a, **kw): return False

    # actions — return True only for buttons we deliberately want to
    # exercise (identified by their `key=`), so we don't accidentally
    # trigger unrelated buttons like sidebar Logout (which has no key
    # and would call st.rerun() before we ever reach the tabs).
    _TRIGGER_KEYS = {
        "btn_ambiguity", "btn_run_sim_study", "btn_run_rob_sweep", "btn_run_blind",
        "btn_repro_exist", "btn_null_exist", "btn_final_decision",
        "btn_robustness_real",
    }
    # Forms whose submit button we want to fire — identified by the
    # form's own name (the string passed to st.form(...)), tracked via
    # _form_stack. Only 'add_ref' (library upload) is enabled here;
    # 'add_user'/'change_pw' stay untriggered.
    _TRIGGER_FORMS = {"add_ref"}
    def button(self, *a, **kw):
        return kw.get("key") in self._TRIGGER_KEYS
    def form_submit_button(self, *a, **kw):
        current_form = self._form_stack[-1] if self._form_stack else None
        return current_form in self._TRIGGER_FORMS
    def form(self, name, **kw):
        return _FormCtx(self, name)

    def stop(self):
        raise _StopExecution()
    def rerun(self):
        raise _StopExecution()

    def cache_resource(self, *a, **kw):
        # dipakai sebagai @st.cache_resource atau @st.cache_resource(ttl=600)
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        def deco(fn):
            return fn
        return deco
    def cache_data(self, *a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        def deco(fn):
            return fn
        return deco


class _StopExecution(BaseException):
    """Simulates Streamlit's internal rerun/stop control-flow signal.
    MUST subclass BaseException (not Exception): app.py has ordinary
    `except Exception as e:` blocks around its own logic (e.g. the
    library-upload flow), and a real Streamlit rerun signal is never
    swallowed by those — it must always propagate up and stop the
    script, exactly like SystemExit/KeyboardInterrupt do."""
    pass


st_stub = _StStub()
st_stub.sidebar = _Ctx()
sys.modules["streamlit"] = st_stub

# CSV palsu untuk menguji alur "Tambah spektra acuan" (form add_ref) —
# 2 kolom wavenumber,absorbance seperti yang diminta app.py.
_FAKE_REF_SPECTRUM_CSV = "wavenumber,absorbance\n" + "\n".join(
    f"{wn:.1f},{np.exp(-((wn-1650)**2)/(2*150**2)):.6f}"
    for wn in np.linspace(400, 4000, 60)
)


# ═══════════════ Stub: plotly ═══════════════
plotly_mod = types.ModuleType("plotly")
go_mod = types.ModuleType("plotly.graph_objects")
express_mod = types.ModuleType("plotly.express")
subplots_mod = types.ModuleType("plotly.subplots")


class _FigStub:
    def __init__(self, *a, **kw): pass
    def add_trace(self, *a, **kw): return self
    def add_vrect(self, *a, **kw): return self
    def add_hline(self, *a, **kw): return self
    def add_vline(self, *a, **kw): return self
    def update_layout(self, *a, **kw): return self
    def update_xaxes(self, *a, **kw): return self
    def update_yaxes(self, *a, **kw): return self


def _trace_stub(*a, **kw): return object()


go_mod.Figure = _FigStub
go_mod.Scatter = _trace_stub
go_mod.Bar = _trace_stub
go_mod.Heatmap = _trace_stub
go_mod.Contour = _trace_stub

express_mod.colors = types.SimpleNamespace(
    qualitative=types.SimpleNamespace(Pastel=["#aaa", "#bbb", "#ccc", "#ddd", "#eee"])
)


def _make_subplots(*a, **kw): return _FigStub()


subplots_mod.make_subplots = _make_subplots

plotly_mod.graph_objects = go_mod
plotly_mod.express = express_mod
plotly_mod.subplots = subplots_mod
sys.modules["plotly"] = plotly_mod
sys.modules["plotly.graph_objects"] = go_mod
sys.modules["plotly.express"] = express_mod
sys.modules["plotly.subplots"] = subplots_mod


# ═══════════════ Stub: httpx (hanya exception class yang dipakai) ═══════════════
httpx_mod = types.ModuleType("httpx")


class TransportError(Exception):
    pass


httpx_mod.TransportError = TransportError
sys.modules["httpx"] = httpx_mod


# ═══════════════ Stub: supabase (stateful — persist across .table() calls) ═══════════════
supabase_mod = types.ModuleType("supabase")


class _FakeResp:
    def __init__(self, data=None, count=0):
        self.data = data
        self.count = count


class _FakeTable:
    """Tidak lagi stateless: baris insert/seed disimpan di
    `store` (dict {table_name: {id: row}}) yang di-share oleh SATU
    _FakeClient, supaya SELECT setelah INSERT/seed benar-benar
    mengembalikan data — dibutuhkan supaya get_all_spectra_for_matching()
    /get_spectrum_by_id() tidak selalu kosong seperti stub semula."""
    _next_id = [1000]

    def __init__(self, name, store):
        self.name = name
        self.store = store
        self.store.setdefault(name, {})
        self._pending_insert = None
        self._filters = []       # list of (col, val) dari .eq()
        self._want_single = False

    def select(self, *a, **kw): return self
    def insert(self, payload, *a, **kw):
        self._pending_insert = payload
        return self
    def update(self, *a, **kw): return self
    def upsert(self, *a, **kw): return self
    def delete(self, *a, **kw): return self
    def eq(self, col, val, *a, **kw):
        self._filters.append((col, val))
        return self
    def order(self, *a, **kw): return self
    def limit(self, *a, **kw): return self
    def single(self, *a, **kw):
        self._want_single = True
        return self

    def _matching_rows(self):
        rows = list(self.store[self.name].values())
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]
        return rows

    def execute(self, *a, **kw):
        # INSERT: simpan permanen ke store (bukan cuma echo transien),
        # supaya SELECT berikutnya (mis. get_all_spectra_for_matching()
        # dipanggil lagi setelah add_ref) benar-benar melihatnya.
        if self._pending_insert is not None:
            row = dict(self._pending_insert)
            new_id = _FakeTable._next_id[0]
            _FakeTable._next_id[0] += 1
            row["id"] = new_id
            self.store[self.name][new_id] = row
            self._pending_insert = None
            return _FakeResp(data=[row])

        # DELETE (self._filters ada, tapi tidak ada select/pending_insert
        # dibedakan cukup kasar di sini — cukup untuk kebutuhan harness ini,
        # bukan tiruan penuh perilaku Supabase delete/update)
        rows = self._matching_rows()
        if self._want_single:
            return _FakeResp(data=(rows[0] if rows else None))
        return _FakeResp(data=rows)


class _FakeClient:
    def __init__(self, store):
        self.store = store
    def table(self, name): return _FakeTable(name, self.store)
    def rpc(self, *a, **kw): return _FakeTable("_rpc", self.store)


# Store bersama SATU instance (persist antar .table() call, sepanjang
# proses harness ini berjalan) — di-seed dengan 1 spektrum referensi
# palsu yang SAMA PERSIS dengan komponen C1 (S_fake[0]) di grid wn_fake
# yang sama, supaya pearson/cosine yang dihitung expander identity
# benar-benar mendekati 1.0 (kasus "match kuat"), bukan angka acak.
_fake_db_store = {}


def create_client(url, key): return _FakeClient(_fake_db_store)


supabase_mod.create_client = create_client
supabase_mod.Client = _FakeClient
sys.modules["supabase"] = supabase_mod


# ═══════════════ Pre-populate a fake logged-in admin session ═══════════════
ss = st_stub.session_state
ss["logged_in"] = True
ss["role"] = "admin"
ss["username"] = "admin"
ss["display_name"] = "Administrator"
ss["lang"] = "id"

n_wn = 250
wn_fake = np.linspace(400, 4000, n_wn)
n_samples_fake = 10
n_comp_fake = 3
rng = np.random.default_rng(0)
C_fake = rng.dirichlet(np.ones(n_comp_fake), size=n_samples_fake)
S_fake = np.abs(rng.normal(size=(n_comp_fake, n_wn))) + 0.1

ss["wavenumber"] = wn_fake
ss["spectra"] = rng.normal(size=(n_wn, n_samples_fake)) + 1.0  # (n_wavenumber x n_sampel), wajib utk guard tab_mcr

# Seed 1 entri "library" palsu ke fake DB — SAMA PERSIS dengan komponen
# C1 (S_fake[0]) pada grid wn_fake yang sama, supaya
# get_all_spectra_for_matching() tidak kosong dan expander identity
# menghitung pearson/cosine NYATA (harusnya ~1.0 utk C1: kasus "match
# kuat"), bukan cuma menembus cabang "belum ada referensi".
_fake_db_store.setdefault("sv_spectra", {})
_fake_db_store["sv_spectra"][1] = {
    "id": 1,
    "name": "Test Reference C1 (harness)",
    "category": "test",
    "wavenumber": json.dumps(wn_fake.tolist()),
    "spectrum": json.dumps(S_fake[0].tolist()),
}
ss["mcr_C"] = C_fake
ss["mcr_S"] = S_fake
ss["mcr_lof"] = [50.0, 20.0, 5.0, 2.0]
ss["mcr_r2"] = 0.98
ss["mcr_ncomp"] = n_comp_fake
ss["mcr_diag"] = {
    "rmse": 0.01,
    "lof_final": 2.0,
    "converged": True,
    "constraints_used": {"s_nonneg": True, "closure": False},
    "nnv_scores": [0.1, 0.2, 0.15],
    "ev_per_comp": [60.0, 25.0, 8.0],
    "lof_per_sample": [2.0] * n_samples_fake,
}
ss["spec_names"] = [f"S{i+1}" for i in range(n_samples_fake)]

print("=== Executing app.py end-to-end with full stub harness ===")
try:
    with open("app.py") as f:
        src = f.read()
    exec(compile(src, "app.py", "exec"), {"__name__": "__main__"})
    print("=== Finished WITHOUT uncaught exception ===")
except _StopExecution:
    print("=== Hit st.stop()/st.rerun() — showing where: ===")
    traceback.print_exc()
except Exception:
    print("=== UNCAUGHT EXCEPTION ===")
    traceback.print_exc()
