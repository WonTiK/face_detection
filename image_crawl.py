# coding: utf-8
from icrawler.builtin import GoogleImageCrawler

for i in range(15, 22):
    str_1 = '%d_people' % i
    # str_2 = '%d_woman' % i
    str_3 = '%d 단체사진' % i
    # str_4 = '%d살 여성' % i
    google_crawler_1 = GoogleImageCrawler(parser_threads = 2, downloader_threads = 4,
            storage = {'root_dir': str_1})
    # google_crawler_2 = GoogleImageCrawler(parser_threads = 2, downloader_threads = 4,
            # storage = {'root_dir': str_2})

    google_crawler_1.crawl(keyword = str_3, max_num = 300,
        date_min = None, date_max = None,
        min_size = (600, 600), max_size = None)
    # google_crawler_2.crawl(keyword = str_4, max_num = 700,
        # date_min = None, date_max = None,
        # min_size = (300, 300), max_size = None)