function RX_data_extract(RxType)

    if strcmp(RxType, "802_11 Framed")
        RX_802_11_Framed()
    elseif strcmp(RxType, "Raw Data")
        RX_Unframed()
    end

end
