<?php if (!defined('BASEPATH')) exit('No direct script access allowed');

use simplehtmldom\HtmlWeb;

class Scrapnews
{
    private function curl_exec($url)
    {
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, true);  // Disable SSL verification
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_REFERER, 'https://scrappernews.test/');
        curl_setopt($ch, CURLOPT_URL, $url);
        $result = curl_exec($ch);
        curl_close($ch);

        return json_decode($result);
    }

    public function news_all($query = ['Saham BBCA'])
    {
        $filtered = false;
        foreach ($query as $q) {
            for ($i = 1; $i <= 10; $i++) {
                $json = $this->curl_exec("https://www.googleapis.com/customsearch/v1?key=AIzaSyDjVXF8DZ1eZ0_AS0gHHli-C2gT9ecYTjM&cx=ab57596f2bc5fe55a&q=" . str_replace(' ', '+', $q) . "&start=" . $i);
                print_r($json);
                if (is_object($json)) {
                    foreach ($json->items as $item) {
                        if (isset($item->pagemap->metatags[0]->{'og:url'}) and strpos($item->pagemap->metatags[0]->{'og:url'}, "/tag/") === false) {
                            $filtered[] = [
                                'title' => $item->title,
                                'describe' => $item->pagemap->metatags[0]->{'og:description'},
                                'author' => isset($item->pagemap->metatags[0]->content_author) ? $item->pagemap->metatags[0]->content_author : ' ',
                                'editor' => isset($item->pagemap->metatags[0]->content_editor) ? $item->pagemap->metatags[0]->content_editor : ' ',
                                'url' => $item->pagemap->metatags[0]->{'og:url'},
                                'source' => $item->displayLink,
                                'published' => isset($item->pagemap->metatags[0]->{'article:published_time'}) ? $item->pagemap->metatags[0]->{'article:published_time'} : ' ',
                                'image' => $item->pagemap->metatags[0]->{'og:image'},
                            ];
                        }
                    }
                }
            }
        }
        return $filtered;
    }

    public function kompasdotcom($query = ['URL'])
    {
        $client = new HtmlWeb();
        $result = false;
        foreach ($query as $q) {
            $html = $client->load($query); // <-- link url news
            // Returns the page title
            $result = [
                'title' => $html->find('title', 0)->plaintext,
                'describe' => $html->find('meta[name=description]', 0)->getAttribute('content'),
                'content' => $html->find('.read__content', 0)->plaintext,
                'author' => $html->find('meta[name=content_author]', 0)->getAttribute('content'),
                'editor' => $html->find('meta[name=content_editor]', 0)->getAttribute('content'),
                'url' => $html->find('meta[property=og:url]', 0)->getAttribute('content'),
                'source' => 'kompas.com',
                'published' => $html->find('meta[name=content_PublishedDate]', 0)->getAttribute('content'),
                'image' => $html->find('meta[property=og:image]', 0)->getAttribute('content'),
            ];
        }

        return $result;
    }
}
