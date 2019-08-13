import scrapy

class RedditSpider(scrapy.Spider):
	name = 'reddits'

	allowed_domains = ['reddit.com']

	start_urls = [
		'https://www.reddit.com/r/funny/'
	]

	