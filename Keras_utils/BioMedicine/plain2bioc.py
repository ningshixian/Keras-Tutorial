#!/usr/bin/env python3

"""Convert plain-text standoff ChemDNER text and annotation files into BioC XML."""

import os
import sys

from datetime import date
from xml.sax.saxutils import escape

__author__ = "Florian Leitner"
__version__ = "1.0"


def main(abstracts_path, annotations_path, keyfile):
    source = os.path.splitext(os.path.basename(abstracts_path))[0]
    today = str(date.today()).replace('-', '')
    print('<?xml version="1.0" encoding="%s"?>' % sys.getdefaultencoding())
    print('<!DOCTYPE collection SYSTEM "BioC.dtd">')
    print('<collection>\n  <source>%s</source>\n  <date>%s</date>\n  <key>%s</key>' % (
        source, today, os.path.basename(keyfile)
    ))

    with open(annotations_path) as annotations_stream:
        annotations = dict(annotations_iter(annotations_stream))

    with open(abstracts_path) as abstracts_stream:
        for pmid, title, abstract in abstract_iter(abstracts_stream):
            title_anns, abstract_anns = annotations[pmid] if pmid in annotations else ([], [])
            print('  <document><id>%s</id>' % pmid)
            passage("title", title, title_anns)
            passage("abstract", abstract, abstract_anns)
            print('  </document>')

    print('</collection>')


def abstract_iter(stream):
    for line in stream:
        yield line.strip().split('\t')


def annotations_iter(stream):
    current_pmid, title_anns, abstract_anns = None, None, None
    lno = 0

    try:
        for lno, line in enumerate(stream, 1):
            pmid, section, start, end, text, infon = line.strip().split('\t')
            ann = (infon, int(start), int(end), text)

            if pmid != current_pmid:
                if current_pmid is not None:
                    yield current_pmid, (title_anns, abstract_anns)

                current_pmid, title_anns, abstract_anns = pmid, [], []

            if section == 'A':
                abstract_anns.append(ann)
            elif section == 'T':
                title_anns.append(ann)
            else:
                raise RuntimeError('unknown annotation section "%s"' % section)
    except UnicodeDecodeError as e:
        raise RuntimeError('encoding error in %s at line %d: %s' % (stream.name, lno, str(e)))


def passage(name, text, annotations):
    passage = '    <passage>\n      <infon key="type">%s</infon>\n      <offset>0</offset>'
    print(passage % name)
    print('      <text>%s</text>' % escape(text))

    for data in annotations:
        annotation(*data)

    print('    </passage>')


def annotation(infon, start, end, text):
    print('      <annotation>')
    print('        <infon key="class">%s</infon>' % infon)
    print('        <location offset="%d" length="%d" />' % (start, end - start))
    print('        <text>%s</text>' % escape(text))
    print('      </annotation>')


if __name__ == '__main__':
    name = os.path.basename(sys.argv[0])

    if '-h' in sys.argv or '--help' in sys.argv:
        print('usage: %s [-h|--help] ABSTRACTS.tsv ANNOTATIONS.tsv INFONS.key\n' % name)
        print(__doc__)
        sys.exit(0)

    if len(sys.argv) != 4:
        print('wrong number of arguments\n', file=sys.stderr)
        print('usage: %s [-h|--help] ABSTRACTS.tsv ANNOTATIONS.tsv INFONS.key' % name,
              file=sys.stderr)
        sys.exit(2)

    try:
        main(*sys.argv[1:])
        sys.exit(0)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
