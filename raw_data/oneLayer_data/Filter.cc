int Filter()
{
    TFile *f = new TFile("efficiency.root");

    TH2D *h = (TH2D *)f->Get("colz");
    TH2D *h_out = (TH2D *)h->Clone("h_out");

    // first fill the zero bins
    for (int i = 1; i <= h->GetNbinsX(); i++)
    {
        for (int j = 1; j <= h->GetNbinsY(); j++)
        {
            if (h->GetBinContent(i, j) == 0)
            {
                double val = h->GetBinContent(i, j - 1) + h->GetBinContent(i, j + 1);
                h_out->SetBinContent(i, j, val / 2.);
            }
        }
    }

    // then filter
    TH2D *h_out2 = (TH2D *)h_out->Clone("h_out2");
    for (int i = 2; i <= h_out->GetNbinsX() - 1; i++)
    {
        for (int j = 2; j <= h_out->GetNbinsY() - 1; j++)
        {
            double val =
                h_out->GetBinContent(i, j - 1) + h_out->GetBinContent(i, j + 1) + 2 * h_out->GetBinContent(i, j);
            h_out2->SetBinContent(i, j, val / 4.);
        }
    }

    // find the max bins
    vector<double> x, y;
    for (int i = 1; i <= h_out2->GetNbinsX(); i++)
    {
        double max = 0;
        int max_bin = -1;
        for (int j = 1; j <= h_out2->GetNbinsY(); j++)
        {
            if (h_out2->GetBinContent(i, j) > max)
            {
                max = h_out2->GetBinContent(i, j);
                max_bin = int(h_out2->GetYaxis()->GetBinCenter(j));
            }
        }
        x.push_back(h_out2->GetXaxis()->GetBinCenter(i));
        y.push_back(max_bin);
    }
    TGraph *g = new TGraph(x.size(), &x[0], &y[0]);

    TFile *fout = new TFile("efficiency_filtered.root", "RECREATE");
    h_out->Write();
    h_out2->Write();
    g->Write();
    fout->Close();

    return 0;
}