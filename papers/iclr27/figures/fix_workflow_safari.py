"""Paper Fig. 1 Safari fix. Usage (needs pikepdf and pymupdf, not in the mace env):
    python fix_workflow_safari.py Workflow.pdf Workflow_fixed.pdf

Remove every soft mask and nested transparency group from Workflow.pdf (paper Fig. 1),
which Safari/PDFKit renders cropped or not at all.

- The page paints the whole drawing (form 52) through an alpha soft mask whose group is an
  opaque rectangle over the full BBox: a no-op, dropped together with form 52's /Group.
- The "latent space" disc (form 38, painted at ca 0.4) is an image clipped by a soft mask
  plus a tinted disc and a dashed outline, all in nested transparency groups. It is
  replaced by one RGB image + /SMask, obtained by rendering the element over black and
  over white (alpha = 1 - (white - black), colour = black / alpha).
Everything else (text, paths, other images, constant opacities) is untouched."""
import sys
import zlib
import numpy as np
import pikepdf
import pymupdf

SRC, DST = sys.argv[1], sys.argv[2]
SCALE = 4  # px per pt (the source raster is 2 px/pt)
BOX = (530.0, 590.0, 786.0, 846.0)  # x0 y0 x1 y1 in form-52 space, around the disc


def render(bg):
    """Render only `/E86 gs /X2 Do` over a solid background, cropped to BOX."""
    pdf = pikepdf.open(SRC)
    f52 = pdf.get_object((52, 0))
    f52.write(f'q {bg} {bg} {bg} rg -10000 -10000 30000 30000 re f Q q /E86 gs /X2 Do Q'.encode())
    pg = pdf.pages[0]
    pg.Contents = pdf.make_stream(b'/C1 CS /C1 cs q /X1 Do Q')
    x0, y0, x1, y1 = BOX
    pg.MediaBox = pikepdf.Array([x0, y0, x1, y1])
    for k in ('/CropBox', '/TrimBox', '/BleedBox', '/ArtBox'):
        if k in pg:
            del pg[k]
    doc = pymupdf.open('pdf', pdf_bytes(pdf))
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False,
                            colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).astype(float) / 255


def pdf_bytes(pdf):
    import io
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


black, white = render(0), render(1)
alpha = np.clip(1 - (white - black).mean(axis=2), 0, 1)
rgb = np.where(alpha[..., None] > 1e-3, black / np.maximum(alpha[..., None], 1e-3), 1.0)
rgb = np.clip(rgb, 0, 1)
h, w = alpha.shape
print(f'disc raster {w}x{h}, alpha range {alpha.min():.3f}-{alpha.max():.3f}')

pdf = pikepdf.open(SRC)
# uncompressed here; pdf.save(compress_streams=True) applies Flate
smask = pikepdf.Stream(pdf, (alpha * 255).round().astype(np.uint8).tobytes(),
                       Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=w,
                       Height=h, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8)
img = pikepdf.Stream(pdf, (rgb * 255).round().astype(np.uint8).tobytes(),
                     Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=w,
                     Height=h, ColorSpace=pikepdf.Name.DeviceRGB, BitsPerComponent=8,
                     SMask=smask)

f52 = pdf.get_object((52, 0))
x0, y0, x1, y1 = BOX
old = b'q\n/E86 gs\n/X2 Do\nQ\n'
data = f52.read_bytes()
assert data.count(old) == 1
new = f'q\n{x1 - x0:.4f} 0 0 {y1 - y0:.4f} {x0:.4f} {y0:.4f} cm\n/XDisc Do\nQ\n'.encode()
f52.write(data.replace(old, new))
del f52.Resources.XObject['/X2']
f52.Resources.XObject['/XDisc'] = img
del f52['/Group']

pg = pdf.pages[0]
content = pg.Contents.read_bytes()
assert content.count(b'/E1 gs\n') == 1
pg.Contents = pdf.make_stream(content.replace(b'/E1 gs\n', b''))
del pg.Resources.ExtGState['/E1']
pdf.remove_unreferenced_resources()
pdf.save(DST, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.preserve)
