<?php
defined('BASEPATH') or exit('No direct script access allowed');

class Scraptest extends CI_Controller
{
	private function download_send_headers($filename)
	{
		$now = gmdate("D, d M Y H:i:s");
		header("Expires: Tue, 03 Jul 2001 06:00:00 GMT");
		header("Cache-Control: max-age=0, no-cache, must-revalidate, proxy-revalidate");
		header("Last-Modified: {$now} GMT");

		// force download  
		header("Content-Type: application/force-download");
		header("Content-Type: application/octet-stream");
		header("Content-Type: application/download");

		// disposition / encoding on response body
		header("Content-Disposition: attachment;filename={$filename}");
		header("Content-Transfer-Encoding: binary");
	}

	private function array2csv(array &$array)
	{
		if (count($array) == 0) {
			return null;
		}
		ob_start();
		$df = fopen("php://output", 'w');
		fputcsv($df, array_keys(reset($array)));
		foreach ($array as $row) {
			fputcsv($df, $row);
		}
		fclose($df);
		return ob_get_clean();
	}

	public function index()
	{
		$this->load->library('Scrapnews');

		$result = $this->scrapnews->news_all(['Saham BBCA', 'Bank Central Asia (BCA)'], 'all');
		if ($result != false) {
			$this->download_send_headers("data_export_" . date("Y-m-d") . ".csv");
			echo $this->array2csv($result);
			die();
		}
		//$this->scrapnews->kompasdotcom('https://money.kompas.com/read/2021/08/18/160112426/ihsg-ditutup-naik-tembus-level-6100-asing-borong-saham-bbca');
	}
}
