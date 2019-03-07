from putils.data import gen_and_save_data

if __name__ == '__main__':
    gen_and_save_data(
        'D:/ACDC_LUNG_HISTOPATHOLOGY/data/', 'D:/ACDC_LUNG_try2/data/pickle/',
        135000, (256, 256), 0
    )
