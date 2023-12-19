from PIL import Image, ImageFont, ImageDraw
import textwrap

class VideoSubtitle:
    def __init__(self, fc, font_path, font_size, image_size, italic=False, background_color=(120, 20, 20), text_color=(255, 255, 0)):
        if italic:
            font_path = font_path.replace('.ttf', '-Italic.ttf')  # Adjust based on your font file naming
        self.font = ImageFont.truetype(font_path, font_size)
        self.sz = image_size
        self.background_color = background_color
        self.text_color = text_color

    def get_text_dimensions(self, text_string):
        ascent, descent = self.font.getmetrics()
        text_width = self.font.getmask(text_string).getbbox()[2]
        text_height = self.font.getmask(text_string).getbbox()[3] + descent
        return (text_width, text_height)

    def estimate_chars_per_line(self, max_width):
        sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        total_width, _ = self.get_text_dimensions(sample_text)
        average_char_width = total_width / len(sample_text)
        return int(max_width / average_char_width)

    def add_subtitles(self, text):
        img = Image.new("RGBA", self.sz, self.background_color)
        draw = ImageDraw.Draw(img)

        max_text_width = self.estimate_chars_per_line(self.sz[0] * fc[0])
        lines = textwrap.wrap(text, width=max_text_width)
        y_text = self.sz[1] * fc[1]

        for line in lines:
            width, height = self.get_text_dimensions(line)
            x_text = (self.sz[0] - width) / 2
            draw.text((x_text, y_text), line, font=self.font, fill=self.text_color)
            y_text += height

        return img

# Usage
fc = (0.7, 0.8)
subtitle = VideoSubtitle(fc, "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-RI.ttf", 20, (1024, 760), italic=False)
text = "We have no need of other worlds. We need mirrors. We don't know what to do with other worlds. A single world, our own, suffices us; but we can't accept it for what it is"
img = subtitle.add_subtitles(text)
img.show()
