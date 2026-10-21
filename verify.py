"""Integration checks: start http.server on port 4174 first."""
from pathlib import Path
import ast, re, json
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set()
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        if tag in ('a','script','link'):
            target=attrs.get('href',attrs.get('src',''))
            if target: self.links.append(target)

pages=list((ROOT/'docs').glob('*.html'))
assert len(pages)==13
parsers={}
for page in pages:
    p=Links(); p.feed(page.read_text(encoding='utf-8')); parsers[page.name]=p
for name,p in parsers.items():
    for link in p.links:
        if link.startswith(('http:','https:')) or link in ('/', '/pytorch-apprentice/', '/ai-era-reader/'): continue
        filename,_,anchor=link.partition('#')
        filename=filename or name
        assert (ROOT/'docs'/filename).exists(), (name,link)
        if anchor and filename in parsers: assert anchor in parsers[filename].ids,(name,link)
for path in (ROOT/'chapters').glob('*.md'):
    text=path.read_text(encoding='utf-8')
    assert text.count('```')%2==0
    for code in re.findall(r'```python\n(.*?)```',text,re.S):
        ast.parse(code,filename=str(path))
for entry in json.loads((ROOT/'docs/search.json').read_text(encoding='utf-8')):
    filename,_,anchor=entry['url'].partition('#')
    assert not anchor or anchor in parsers[filename].ids
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    for width in (390,1440):
        context=browser.new_context(viewport={'width':width,'height':900},permissions=['clipboard-read','clipboard-write'])
        page=context.new_page(); errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        for name in parsers:
            response=page.goto('http://127.0.0.1:4174/'+name)
            assert response.status==200
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,name)
        page.goto('http://127.0.0.1:4174/chapter-10.html')
        assert page.locator('math').count()>30
        page.locator('#theme').click()
        assert page.locator('html').get_attribute('data-theme')=='dark'
        page.locator('#larger').click()
        assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--reader-size')").strip()=='19px'
        page.reload()
        assert page.locator('html').get_attribute('data-theme')=='dark'
        assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--reader-size')").strip()=='19px'
        copy=page.locator('.copy-code').first; copy.click()
        expect(copy).to_have_text('已复制')
        assert page.evaluate('navigator.clipboard.readText()')
        page.evaluate("scrollTo({top:(document.documentElement.scrollHeight-innerHeight)*0.5,behavior:'instant'})")
        page.wait_for_timeout(400)
        page.reload(); page.wait_for_timeout(300)
        assert .45 < page.evaluate('scrollY/(document.documentElement.scrollHeight-innerHeight)') < .55
        page.goto('http://127.0.0.1:4174/index.html')
        assert page.locator('#resume').is_visible()
        page.locator('#search-open').click(); page.locator('#query').fill('zero_grad')
        page.wait_for_function("document.querySelectorAll('.search-result').length>0")
        assert page.locator('mark').count()>0
        page.keyboard.press('Escape'); page.locator('#search-dialog').wait_for(state='hidden')
        page.locator('#theme').click()
        (ROOT/'test-results').mkdir(exist_ok=True)
        page.screenshot(path=str(ROOT/'test-results'/f'home-{width}.png'),full_page=True)
        page.goto('http://127.0.0.1:4174/chapter-10.html')
        page.evaluate("scrollTo({top:0,behavior:'instant'})"); page.screenshot(path=str(ROOT/'test-results'/f'chapter-{width}.png'))
        assert not errors,errors
        context.close()
    browser.close()
print('PASS: 13 pages, local files/anchors, Python syntax, mobile/desktop layouts, math, theme/font persistence, clipboard, progress restore and full-text search.')
