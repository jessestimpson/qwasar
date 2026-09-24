import AppKit
import CoreText

/// The menu bar icon: a large Q, plain when all is well, with a small status
/// dot at its upper right when there is something else to say.
///
/// Drawn rather than taken from SF Symbols, which only offer the letter inside
/// a circle, at half the size.  The Q is a glyph from the system font, so it
/// matches the menu bar's own text.
///
/// It is two pieces.  The Q is a template image, so the menu bar tints it the
/// way it tints every other item -- light, dark, wallpaper-tinted, highlighted.
/// The dot cannot be part of a template (templates are one colour), so it is a
/// layer on the button.  When the dot shows, the Q is swapped for a copy with
/// a ring cut out around it, the way a notification badge sits on an icon, so
/// the dot reads as a dot rather than a smudge on the letter's edge.
enum StatusIcon {
    enum Dot {
        case none, amber, red

        var color: NSColor? {
            switch self {
            case .none:  return nil
            case .amber: return .systemOrange
            case .red:   return .systemRed
            }
        }
    }

    struct Glyph {
        let plain: NSImage       // template: the Q alone
        let badged: NSImage      // template: the Q with the dot's ring cut out
        let dotCenter: CGPoint   // in the images' coordinates, y up
        let dotRadius: CGFloat
    }

    /// `height` is the menu bar's thickness; the Q's cap height is most of it.
    static func glyph(height: CGFloat = NSStatusBar.system.thickness) -> Glyph {
        let probe = NSFont.systemFont(ofSize: 100, weight: .bold)
        let size = (height * 0.66) / (probe.capHeight / 100)
        let font = NSFont.systemFont(ofSize: size, weight: .bold) as CTFont

        var chars: [UniChar] = Array("QO".utf16)
        var glyphs = [CGGlyph](repeating: 0, count: 2)
        CTFontGetGlyphsForCharacters(font, &chars, &glyphs, 2)
        guard let q = CTFontCreatePathForGlyph(font, glyphs[0], nil),
              let o = CTFontCreatePathForGlyph(font, glyphs[1], nil) else {
            let empty = NSImage(size: NSSize(width: height, height: height))
            return Glyph(plain: empty, badged: empty,
                         dotCenter: CGPoint(x: height, y: height), dotRadius: 0)
        }
        let qBox = q.boundingBoxOfPath
        let bowl = o.boundingBoxOfPath       // the Q's bowl, without the tail

        // The bowl is centred vertically; the tail hangs below it, as it would
        // in a line of text.  The dot sits on the bowl's upper-right corner,
        // and the image is widened on the right to hold it.
        let r = max(2.5, height * 0.13)
        let gap = max(1, r * 0.45)
        let pad: CGFloat = 1
        let dx = pad - qBox.minX
        let dy = (height - bowl.height) / 2 - bowl.minY
        let center = CGPoint(x: dx + bowl.maxX - r * 0.2, y: dy + bowl.maxY - r * 0.2)
        let width = ceil(max(dx + qBox.maxX, center.x + r) + pad)

        func draw(cutout: Bool) -> NSImage {
            let image = NSImage(size: NSSize(width: width, height: height), flipped: false) { _ in
                guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
                ctx.saveGState()
                ctx.translateBy(x: dx, y: dy)
                ctx.setFillColor(.black)     // a template's colour is ignored; alpha is what counts
                ctx.addPath(q)
                ctx.fillPath()
                ctx.restoreGState()
                if cutout {
                    let k = r + gap
                    ctx.setBlendMode(.clear)
                    ctx.fillEllipse(in: CGRect(x: center.x - k, y: center.y - k,
                                               width: k * 2, height: k * 2))
                }
                return true
            }
            image.isTemplate = true
            return image
        }
        return Glyph(plain: draw(cutout: false), badged: draw(cutout: true),
                     dotCenter: center, dotRadius: r)
    }
}

/// The status dot: a filled circle on its own layer, so its colour survives
/// the menu bar's tinting and it can pulse without touching the Q.
final class DotView: NSView {
    var color: NSColor? {
        didSet { layer?.backgroundColor = color?.cgColor; isHidden = color == nil }
    }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.cornerRadius = frame.width / 2
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // The button underneath takes every click.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}
