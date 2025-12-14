import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Highlight:
    title: str
    author: str
    metadata: str
    content: str
    
    @property
    def id(self) -> str:
        """Generate a unique ID based on content hash."""
        import hashlib
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()

class ClippingsParser:
    SEPARATOR = "=========="

    def parse(self, file_path: str) -> List[Highlight]:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        # Split by separator and remove empty entries
        raw_clippings = [c.strip() for c in content.split(self.SEPARATOR) if c.strip()]
        
        highlights = []
        for raw in raw_clippings:
            parsed = self._parse_single(raw)
            if parsed:
                highlights.append(parsed)
                
        return highlights

    def _parse_single(self, raw_clipping: str) -> Optional[Highlight]:
        lines = raw_clipping.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        if len(lines) < 3:
            return None
            
        # Line 1: Title and Author
        # Usually format: Title (Author)
        title_line = lines[0]
        author_match = re.search(r'\((.*?)\)$', title_line)
        if author_match:
            author = author_match.group(1)
            title = title_line[:author_match.start()].strip()
        else:
            author = "Unknown"
            title = title_line

        # Line 2: Metadata
        metadata = lines[1]
        
        # Remaining lines: Content
        content = "\n".join(lines[2:])
        
        return Highlight(
            title=title,
            author=author,
            metadata=metadata,
            content=content
        )

if __name__ == "__main__":
    # Test run
    parser = ClippingsParser()
    highlights = parser.parse("My Clippings.txt")
    for h in highlights:
        print(f"Title: {h.title}")
        print(f"Author: {h.author}")
        print(f"Content: {h.content[:50]}...")
        print("-" * 20)
