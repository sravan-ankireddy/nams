function TX_data_generate(TxType, seed)

    if strcmp(TxType, "802_11 Framed")
        TX_802_11_Framed(seed)
    elseif strcmp(TxType, "Raw Data")
        TX_Unframed()
    end

end
