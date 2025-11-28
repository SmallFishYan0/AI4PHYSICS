double interpolate(TH2D *h, double x, double y)
{
    if (x < 400 || y < 0 || y > 60)
    {
        return 0;
    }
    double xbin = (x - 400) / 50. + 1;
    int xbin_low = xbin;
    int xbin_upp = xbin_low + 1;

    int ymax_bin_low = 0, ymax_bin_upp = 0;
    double ymax_low = 0, ymax_upp = 0;
    for (int i = 1; i < h->GetNbinsY(); i++)
    {
        if (h->GetBinContent(xbin_low, i) > ymax_low)
        {
            ymax_bin_low = i;
            ymax_low = h->GetBinContent(xbin_low, i);
        }
        if (h->GetBinContent(xbin_upp, i) > ymax_upp)
        {
            ymax_bin_upp = i;
            ymax_upp = h->GetBinContent(xbin_upp, i);
        }
    }

    double ymax_bin = (ymax_bin_upp - ymax_bin_low) / 1. * (xbin - xbin_low) + ymax_bin_low;
    double ydiff = y - ymax_bin;
    int ydiff_bin = ydiff;
    int ydiff_bin2 = ydiff > 0 ? ydiff_bin + 1 : ydiff_bin - 1;

    double val_low =
        (h->GetBinContent(xbin_low, ymax_bin_low + ydiff_bin) - h->GetBinContent(xbin_low, ymax_bin_low + ydiff_bin2))
            / 1. * (ydiff - ydiff_bin)
        + h->GetBinContent(xbin_low, ymax_bin_low + ydiff_bin2);
    double val_upp =
        (h->GetBinContent(xbin_upp, ymax_bin_upp + ydiff_bin) - h->GetBinContent(xbin_upp, ymax_bin_upp + ydiff_bin2))
            / 1. * (ydiff - ydiff_bin)
        + h->GetBinContent(xbin_upp, ymax_bin_upp + ydiff_bin2);
    double val = (val_upp - val_low) / 1. * (xbin - xbin_low) + val_low;

    return val;
}

int interpolate()
{
    TFile *fin = new TFile("efficiency_filtered.root");
    TH2D *h = (TH2D *)fin->Get("h_out2");

    TH2D *h1 = new TH2D("h1", "h1", 100, 500, 1000, 100, 0, 50);
    for (int i = 1; i <= h1->GetNbinsX(); i++)
    {
        for (int j = 1; j <= h1->GetNbinsY(); j++)
        {
            double x = h1->GetXaxis()->GetBinCenter(i);
            double y = h1->GetYaxis()->GetBinCenter(j);
            double val = interpolate(h, x, y);
            val = val > 0 ? val : 0;
            h1->SetBinContent(i, j, val);
        }
    }

    TFile *f = new TFile("interpolate.root", "RECREATE");
    h1->Write();
    f->Close();

    return 0;
}