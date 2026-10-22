"""A browser reading view of the live document, with a raw text fallback."""
import html
import re

from fastapi.responses import FileResponse, HTMLResponse, Response


def html_preferred(accept):
    weights={}
    for item in accept.lower().split(','):
        parts=[part.strip() for part in item.split(';')]
        quality=1.0
        for part in parts[1:]:
            if part.startswith('q='):
                try:quality=float(part[2:])
                except ValueError:quality=0
        weights[parts[0]]=quality
    return weights.get('text/html',0)>weights.get('text/plain',weights.get('*/*',0))


def document_response(request, website):
    headers={'Cache-Control':'no-store, max-age=0, must-revalidate','Vary':'Accept'}
    path=website/'llms.txt'
    if not path.is_file():
        return Response('Document unavailable',status_code=404,media_type='text/plain',headers=headers)
    if request.query_params.get('raw')=='1' or not html_preferred(request.headers.get('accept','')):
        return FileResponse(path,media_type='text/plain; charset=utf-8',headers=headers)
    from markdown_it import MarkdownIt
    source=path.read_text()
    # The source intentionally has a plain, unfenced shell command. Preserve
    # those newlines in the browser without editing the machine-readable file.
    chunks=source.split('\n\n')
    chunks=['```text\n'+chunk+'\n```' if chunk.startswith('CUBLAS_WORKSPACE_CONFIG=') else chunk.replace('\n','  \n') if chunk.startswith('Dashboard:') else chunk for chunk in chunks]
    markdown=MarkdownIt('commonmark',{'html':False})
    def text_links(tokens,index,options,env):
        text=tokens[index].content
        inside_link=sum(token.type=='link_open' for token in tokens[:index])-sum(token.type=='link_close' for token in tokens[:index])
        if inside_link:return html.escape(text)
        parts=re.split(r'(https?://[^\s<>]+)',text)
        return ''.join('<a href="'+html.escape(part,quote=True)+'">'+html.escape(part)+'</a>'
                       if part.startswith(('https://','http://')) else html.escape(part) for part in parts)
    markdown.renderer.rules['text']=text_links
    body=markdown.render('\n\n'.join(chunks))
    template=(website/'llms-page.html').read_text()
    return HTMLResponse(template.replace('<!--DOCUMENT-->',body),headers=headers)
