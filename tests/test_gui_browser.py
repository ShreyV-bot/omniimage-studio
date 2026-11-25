"""Drives the timeline workspace in a real browser (mouse drags, keys, inspector). Skipped when Playwright or a
Chrome/Chromium binary is not installed."""
import os
import shutil
import threading
import time

import cv2
import numpy as np
import pytest

pw = pytest.importorskip("playwright.sync_api")
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

from omniimage.gui.server import Studio, StudioServer  # noqa: E402


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    work = tmp_path_factory.mktemp("browser") / "work"
    os.makedirs(work)
    cv2.imwrite(str(work / "a.png"), np.dstack([np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (90, 1))] * 3))
    cv2.imwrite(str(work / "b.png"), np.full((90, 160, 3), (0, 0, 255), np.uint8))
    studio = Studio(workdir=str(work))
    srv = StudioServer(studio, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"]) if CHROME else p.chromium.launch()
        except Exception as e:                                       # no usable browser on this machine
            srv.shutdown()
            pytest.skip(f"no browser available: {e}")
        pg = browser.new_page(viewport={"width": 1500, "height": 900})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
        pg.on("dialog", lambda d: d.accept("beat") if d.type == "prompt" else d.accept())
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/")
        pg.wait_for_selector("#libList .it", state="attached")
        pg.click("#tabTimeline")
        pg.work = work
        yield pg
        browser.close()
    srv.shutdown()


def view(pg):
    return pg.evaluate("j('/api/timeline')")


def clips(pg, t=0):
    return [(c["name"], c["start"], c["duration"]) for c in view(pg)["tracks"][t]["clips"]]


def settle(pg, ms=400):
    pg.wait_for_timeout(ms)


def drag(pg, x0, y0, dx, steps=6):
    pg.mouse.move(x0, y0)
    pg.mouse.down()
    pg.mouse.move(x0 + dx, y0, steps=steps)
    pg.mouse.up()
    settle(pg)


def test_timeline_workspace_end_to_end(page):
    pg = page
    assert pg.locator("#tlpage").is_visible() and not pg.locator("#canvas").is_visible()
    # media dropdown creates the track and places the clip at the playhead
    pg.select_option("#tlmedia", "video:a.png")
    pg.wait_for_selector(".clip.v")
    assert clips(pg) == [("a.png", 0, 48)]
    zoom = pg.evaluate("S.tl.zoom")

    # mouse drag moves the clip by whole frames
    b = pg.locator(".clip.v").bounding_box()
    drag(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, 120)
    assert clips(pg) == [("a.png", round(120 / zoom), 48)]

    # right-edge handle trims the end
    b = pg.locator(".clip.v").bounding_box()
    drag(pg, b["x"] + b["width"] - 3, b["y"] + b["height"] / 2, -12 * zoom)
    assert clips(pg)[0][2] == 36

    # left-edge handle trims the head and advances the source offset
    b = pg.locator(".clip.v").bounding_box()
    drag(pg, b["x"] + 3, b["y"] + b["height"] / 2, 6 * zoom)
    c = view(pg)["tracks"][0]["clips"][0]
    assert c["in_point"] == 6 and c["duration"] == 30

    # ruler click moves the playhead, S splits there, Delete removes the selected half, Ctrl+Z brings it back
    r = pg.locator("#tlruler").bounding_box()
    mid = c["start"] + 15
    pg.mouse.click(r["x"] + mid * zoom, r["y"] + 10)
    settle(pg, 200)
    assert pg.evaluate("S.tl.head") == mid
    pg.keyboard.press("s")
    settle(pg)
    assert [start for _, start, _ in clips(pg)] == [c["start"], mid]
    pg.keyboard.press("Delete")
    settle(pg)
    assert len(clips(pg)) == 1
    pg.keyboard.press("Control+z")
    settle(pg)
    assert len(clips(pg)) == 2

    assert pg.errors == []


def test_transition_effect_and_param_edit_through_inspector(page):
    pg = page
    pg.click("#new")
    settle(pg)
    pg.select_option("#tlmedia", "video:a.png")
    pg.wait_for_selector(".clip.v")
    pg.evaluate("tlSetHead(30)")
    pg.select_option("#tlmedia", "video:b.png")
    settle(pg, 600)
    pg.locator(".clip.v").nth(1).click(position={"x": 40, "y": 20})
    settle(pg)
    pg.locator("#insp select").nth(1).select_option("dissolve")
    pg.click("text=Apply (overlaps the previous clip)")
    settle(pg)
    assert view(pg)["tracks"][0]["clips"][1]["transition"] == {"type": "dissolve", "duration": 12}
    effect = pg.locator('#insp select:has(option[value=""])').last
    effect.select_option("blur_gaussian")
    settle(pg, 700)
    box = pg.locator("#insp .p input[type=number]").last
    box.fill("6")
    box.evaluate("e => e.dispatchEvent(new Event('change'))")
    settle(pg, 600)
    nodes = pg.evaluate("j('/api/timeline/clip?track=0&clip=1')")["graph"]["nodes"]
    assert nodes[-1]["type"] == "blur_gaussian" and nodes[-1]["params"]["radius"] == 6
    pg.click("#insp >> text=Remove")
    settle(pg, 600)
    assert [n["type"] for n in pg.evaluate("j('/api/timeline/clip?track=0&clip=1')")["graph"]["nodes"]] == ["load_image"]
    assert pg.errors == []


def test_demo_has_thumbnails_waveforms_and_playback_advances(page):
    pg = page
    pg.click("#tlDemo")
    pg.wait_for_selector(".clip.a")
    for _ in range(40):
        if pg.evaluate("Object.keys(S.tl.thumbs).length") >= 3 and pg.evaluate("Object.keys(S.tl.peaks).length") >= 3:
            break
        settle(pg, 250)
    assert pg.evaluate("Object.keys(S.tl.thumbs).length") >= 3
    assert pg.evaluate("Object.keys(S.tl.peaks).length") >= 3
    pg.keyboard.press("Home")
    pg.keyboard.press(" ")
    settle(pg, 1200)
    playing_head = pg.evaluate("S.tl.head")
    pg.keyboard.press(" ")
    assert playing_head > 3 and pg.evaluate("S.tl.playing") is False
    assert pg.errors == []


def test_save_open_and_switching_workspaces(page):
    pg = page
    pg.click("#save")
    settle(pg, 500)
    assert os.path.exists(pg.work / "beat.timeline.json")
    pg.click("#new")
    settle(pg)
    assert view(pg)["tracks"] == []
    pg.select_option("#files", "beat.timeline.json")
    settle(pg, 800)
    assert len(view(pg)["tracks"]) >= 2
    pg.click("#tabNodes")
    assert pg.locator("#canvas").is_visible() and not pg.locator("#tlpage").is_visible()
    pg.click("#tabTimeline")
    assert pg.locator("#tlpage").is_visible()
    assert pg.errors == []
