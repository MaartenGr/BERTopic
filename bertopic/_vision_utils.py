
try:
    from PIL import Image
    _has_vision = True
except ImportError:
    _has_vision = False


def open_image(image):
    return Image.open(image)


def get_concat_h_multi_resize(im_list):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    min_height = min(im.height for im in im_list)
    im_list_resize = []
    for im in im_list:
        im.resize((int(im.width * min_height / im.height), min_height), resample=0)
        im_list_resize.append(im)

    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def get_concat_v_multi_resize(im_list):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=0)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def get_concat_tile_resize(im_list_2d):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    im_list_v = [get_concat_h_multi_resize(im_list_h) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v).resize((600, 600))