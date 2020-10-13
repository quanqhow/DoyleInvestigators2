import requests
import re
from bs4 import BeautifulSoup
import string
from collections import namedtuple

'''
Important: Calling `.scrape()` often can and will cause your Colab instance to be banned from querying Gutenberg
For debugging purposes, consider doing the following:
  `responses = [requests.get(url) for url in urls]`
  `scraper = Scraper(urls)`
  `books = scraper.scrape(responses=responses)`
'''


urls = [
    'https://www.gutenberg.org/cache/epub/9902/pg9902.html', # The Middle of Things
    'https://www.gutenberg.org/cache/epub/9807/pg9807.html', # Scarhaven Keep
    'https://www.gutenberg.org/cache/epub/9834/pg9834.html', # The Talleyrand Maxim
    'https://www.gutenberg.org/cache/epub/10373/pg10373.html', # The Middle Temple Murder
    'http://www.gutenberg.org/cache/epub/12239/pg12239.html' # Dead Men's Money
]

Match = namedtuple('Match', ['chapter_number', 'sentence_number', 'matches', 'text'])

class Scraper:
    def __init__(self, urls):
        self.urls = urls

    def scrape(self, *, responses=None):
        books = []
        responses = [requests.get(url) for url in self.urls] if responses is None else responses

        for response in responses:

            positions = [
                         response.text.find("***END"),
                         response.text.find("THE END")
                        ]
            end_pos = min([pos for pos in positions if pos > 0])
            book_html = response.text[:end_pos]

            def search(pattern):
                return re.search(pattern, book_html).group(1)

            title = search('Title: ([^\<]*)')
            author = search('Author: ([^\<]*)')
            release = search('Release Date: ([A-Za-z]*[\s0-9]*?, [0-9]*)')

            soup = BeautifulSoup(book_html)

            chapters = self.find_chapters(soup)

            books.append(Book(author, chapters, release, title))

        return books

    def find_chapters(self, soup):
        chapters = []
        chapter_tags = soup.find_all(
            re.compile("(h2|h3)"),
            string=re.compile("Chapter", re.IGNORECASE)
        )
        for chapter_number, chapter_tag in enumerate(chapter_tags, start=1):
            chapter_text = []
            chapter_title = None
            for i, sibling in enumerate(chapter_tag.next_siblings):
                if i == 1:
                    chapter_title = string.capwords(sibling.text)
                elif sibling.name == 'p':
                    chapter_text.append(sibling.text)
                elif sibling.name == 'h2':
                    break
                else:
                    continue

            chapters.append(
                Chapter(chapter_title, chapter_number, chapter_text)
            )

        return chapters

class Chapter:
    def __init__(self, title, number, text):
        self.title = title
        self.number = number
        self.text = self.clean(text)

    def find(self, pattern):
        all_matches = []
        for sentence_number, sentence in self.sentences:
            matches = list(re.finditer(pattern, sentence))
            if len(matches):
                all_matches.append(
                    Match(self.number, sentence_number, matches, sentence)
                )

        return all_matches

    @staticmethod
    def clean(chapter_text):
        chapter_text = ' '.join(chapter_text)
        chapter_text = chapter_text.replace('\r\n', ' ')
        return chapter_text

    @property
    def sentences(self):
        '''
        Splits the chapter text into sentences. This is a hard task so some decisions need to be made.

        Case 1: (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=[.?!])\s
            (?<!\w\.\w.): Negative lookbehind to prevent splitting on i.e. and e.g.
        Python requires fixed width patterns for lookbehinds, so we have split these
            (?<![A-Z][a-z]\.): Negative lookbehind to prevent splitting on Mr.
            (?<![A-Z][a-z][a-z]\.): Negative lookbehind to prevent splitting on Mrs.
            (?<=[.?!]): Positive lookbehind to make sure we're only splitting after ., ?, or !
            \s: Any white-space character
        Case 2: (?<=[.?!][\"])\s(?=[\"A-Z])
            (?<=[.?!][\"]): Positive lookbehind to make sure we're only splitting after .", ?", or !"
            \s: Any white-space character
            (?=[\"A-Z]): Positive lookahead to make sure we're only splitting before " or a capital letter

        Examples:
            ...a fashion which had become a habit. Miss Penkridge...
                Sentence 1: ...a fashion which had become a habit.
                Sentence 2: Miss Penkridge...
            ...content. "So he did it! Now, I should never have thought it! The last person...
                Sentence 1: ...content.
                Sentence 2: "So he did it!
                Sentence 3: Now, I should never have thought it!
                Sentence 4: The last person...
            ...this sort of stuff?" "Stuff?" demanded Miss Penkridge, who had resumed her...
                Sentence 1: ...this sort of stuff?"
                Sentence 2: "Stuff?" demanded Miss Penkridge, who had resumed her...
        '''
        splits = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=[.?!])\s|(?<=[.?!][\"])\s(?=[\"A-Z])', self.text)
        return zip(range(1, len(splits)+1), splits)

    def __len__(self):
        return len(list(self.sentences))

    def __str__(self):
        return self.title

    def __repr__(self):
        return (
            f'Chapter {self.number}: {self.title}\n'
            f'Content: {self.text[:20]}...{self.text[-20:]}'
        )

class Book:
    def __init__(self, author, chapters, release, title):
        self.author = author
        self.chapters = chapters
        self.release = release
        self.title = title

    def find(self, pattern):
        all_matches = []
        for chapter in self.chapters:
            matches = chapter.find(pattern)
            if len(matches):
                all_matches.extend(matches)

        return all_matches

    def __len__(self):
        return len(self.chapters)

    def __str__(self):
        return self.title

    def __repr__(self):
        return (
            f'{self.title} by {self.author}, '
            f'released in {self.release} and '
            f'contains {len(self)} chapters.'
        )