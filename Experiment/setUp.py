import os
import pygame as pg
import config


main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "images")


# def load_image(name, colorkey=None, scale=1):
#     fullname = os.path.join(data_dir, name)
#     image = pg.image.load(fullname)
#     image = pg.transform(image, (250, 50))

#     if colorkey is not None:
#         if colorkey == -1:
#             colorkey = image.get_at((0, 0))
#         image.set_colorkey(colorkey, pg.RLEACCEL)
#     return image, image.get_rect()

class GUI:

    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((config.BACKGROUND['width'], config.BACKGROUND['height']))
        pg.display.set_caption(config.BACKGROUND['window_name'])
        pg.mouse.set_visible(True)
        self.background = pg.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(config.BACKGROUND['colour'])
        self.state = None
        self.allsprites = None

        # Put Text On The Background, Centered
        if pg.font:
            font = pg.font.Font(None, 20)
            text = font.render("Balance the inverted pendulum", True, (10, 10, 10))
            textpos = text.get_rect(centerx=self.background.get_width() / 2, y=10)
            self.background.blit(text, textpos)

        # Display The Background
        self.screen.blit(self.background, (0, 0))
        pg.display.flip()

    def set_state(self, state):
        self.state = state
    
    def clean_background(self):
        self.screen.blit(self.background, (0, 0))
        pg.display.flip()

    def get_objects(self):
        pendulum = self.state.get_pendulum()
        cart = self.state.get_cart()
        return pendulum, cart

    def draw(self):
        pendulum, cart = self.get_objects()
        self.draw_cart_and_pendulum(pendulum, cart)


    def draw_cart_and_pendulum(self, pendulum, cart):
        allsprites = pg.sprite.RenderPlain((pendulum, cart))
        self.allsprites = allsprites
        allsprites.draw(self.screen)


    def update_scene(self, vel):
        self.state.set_cart_vel(vel)
        self.state.update()
        self.allsprites.update()




