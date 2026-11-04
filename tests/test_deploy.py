"""Deploy = the old deploy_desktop_resume.ps1 layout: resumes/<slug>.tex, build/<slug>.pdf, Desktop copy."""
from resume_agent.deploy import DESKTOP_NAME, GENERATED_MARK, deploy


def _files(tmp_path):
    tex = tmp_path / "out.tex"
    tex.write_text("\documentclass{article}", encoding="utf-8")
    pdf = tmp_path / "out.pdf"
    pdf.write_bytes(b"%PDF-1.5 fake")
    return tex, pdf


def test_deploys_tex_build_and_desktop(tmp_path):
    tex, pdf = _files(tmp_path)
    root, desk = tmp_path / "Resume_Builder", tmp_path / "Desktop"
    d = deploy(tex, pdf, 1, "acme_ai_intern", root=root, desktop=desk)
    assert d.tex == root / "resumes" / "acme_ai_intern.tex"
    assert d.tex.read_text(encoding="utf-8").startswith(GENERATED_MARK)
    assert (root / "build" / "acme_ai_intern.pdf").read_bytes() == pdf.read_bytes()
    assert d.desktop == desk / DESKTOP_NAME and d.desktop.read_bytes() == pdf.read_bytes()


def test_never_overwrites_a_hand_written_resume(tmp_path):
    tex, pdf = _files(tmp_path)
    root = tmp_path / "Resume_Builder"
    (root / "resumes").mkdir(parents=True)
    mine = root / "resumes" / "acme.tex"
    mine.write_text("my hand-written resume", encoding="utf-8")
    d = deploy(tex, pdf, 1, "acme", root=root, desktop=tmp_path / "Desktop")
    assert mine.read_text(encoding="utf-8") == "my hand-written resume"
    assert d.tex.name == "acme_agent.tex"


def test_rerun_overwrites_its_own_file(tmp_path):
    tex, pdf = _files(tmp_path)
    root = tmp_path / "Resume_Builder"
    deploy(tex, pdf, 1, "acme", root=root, desktop=tmp_path / "Desktop")
    tex.write_text("\documentclass{article} % v2", encoding="utf-8")
    d = deploy(tex, pdf, 1, "acme", root=root, desktop=tmp_path / "Desktop")
    assert d.tex.name == "acme.tex" and "v2" in d.tex.read_text(encoding="utf-8")


def test_reuses_the_desktop_files_own_spelling(tmp_path):
    tex, pdf = _files(tmp_path)
    desk = tmp_path / "Desktop"
    desk.mkdir()
    (desk / "om_tilwar_resume.pdf").write_bytes(b"old")
    d = deploy(tex, pdf, 1, "acme", root=tmp_path / "RB", desktop=desk)
    assert d.desktop.name == "om_tilwar_resume.pdf"
    assert [p.name for p in desk.iterdir()] == ["om_tilwar_resume.pdf"]


def test_two_page_resume_is_not_put_on_the_desktop(tmp_path):
    tex, pdf = _files(tmp_path)
    desk = tmp_path / "Desktop"
    d = deploy(tex, pdf, 2, "acme", root=tmp_path / "RB", desktop=desk)
    assert d.desktop is None and "2 pages" in d.skipped
    assert not desk.exists()
    assert d.tex.exists()                                   # the source is still saved for fixing
