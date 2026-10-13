// The app's icon, drawn at build time rather than kept as a binary asset:
// the same bold system-font Q as the menu bar item (StatusIcon.swift), white
// on a deep navy tile with a bright core behind it -- a quasar, near enough.
//
//   AppIcon <dir.iconset>     writes every size iconutil wants
//
// The Makefile runs it and `iconutil -c icns` on the result.  The tile follows
// the macOS icon grid: an 824-point body on a 1024 canvas, corner radius
// ~185, with a soft shadow beneath.

import AppKit
import CoreText

func render(_ px: Int) -> Data? {
    guard let ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    let s = CGFloat(px) / 1024
    ctx.scaleBy(x: s, y: s)

    let body = CGRect(x: 100, y: 100, width: 824, height: 824)
    let tile = CGPath(roundedRect: body, cornerWidth: 185, cornerHeight: 185, transform: nil)

    // Shadow, then the tile's gradient clipped to it.
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 28,
                  color: CGColor(gray: 0, alpha: 0.35))
    ctx.addPath(tile)
    ctx.setFillColor(CGColor(srgbRed: 0.05, green: 0.06, blue: 0.16, alpha: 1))
    ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(tile)
    ctx.clip()
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    let fill = CGGradient(colorsSpace: space, colors: [
        CGColor(srgbRed: 0.19, green: 0.21, blue: 0.52, alpha: 1),   // top: indigo
        CGColor(srgbRed: 0.04, green: 0.05, blue: 0.15, alpha: 1),   // bottom: near black
    ] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(fill, start: CGPoint(x: 512, y: 924), end: CGPoint(x: 512, y: 100), options: [])
    // The core: a soft bright glow where the Q's bowl sits.
    let glow = CGGradient(colorsSpace: space, colors: [
        CGColor(srgbRed: 0.62, green: 0.70, blue: 1.0, alpha: 0.55),
        CGColor(srgbRed: 0.40, green: 0.45, blue: 1.0, alpha: 0.0),
    ] as CFArray, locations: [0, 1])!
    ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 512, y: 540), startRadius: 0,
                           endCenter: CGPoint(x: 512, y: 540), endRadius: 400, options: [])
    ctx.restoreGState()

    // The Q, its bowl (an O's box) centred in the tile, the tail hanging below.
    let probe = NSFont.systemFont(ofSize: 100, weight: .bold)
    let size = 470 / (probe.capHeight / 100)
    let font = NSFont.systemFont(ofSize: size, weight: .bold) as CTFont
    var chars: [UniChar] = Array("QO".utf16)
    var glyphs = [CGGlyph](repeating: 0, count: 2)
    CTFontGetGlyphsForCharacters(font, &chars, &glyphs, 2)
    guard let q = CTFontCreatePathForGlyph(font, glyphs[0], nil),
          let o = CTFontCreatePathForGlyph(font, glyphs[1], nil) else { return nil }
    let bowl = o.boundingBoxOfPath
    ctx.translateBy(x: 512 - bowl.midX, y: 540 - bowl.midY)
    ctx.setShadow(offset: CGSize(width: 0, height: -6), blur: 18,
                  color: CGColor(srgbRed: 0.02, green: 0.03, blue: 0.12, alpha: 0.6))
    ctx.addPath(q)
    ctx.setFillColor(CGColor(gray: 1, alpha: 1))
    ctx.fillPath()

    guard let image = ctx.makeImage() else { return nil }
    return NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
}

let args = CommandLine.arguments
guard args.count == 2 else {
    FileHandle.standardError.write("usage: AppIcon <dir.iconset>\n".data(using: .utf8)!)
    exit(2)
}
let dir = URL(fileURLWithPath: args[1])
try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        guard let png = render(base * scale) else {
            FileHandle.standardError.write("AppIcon: could not draw \(name)\n".data(using: .utf8)!)
            exit(1)
        }
        try png.write(to: dir.appendingPathComponent(name))
    }
}
